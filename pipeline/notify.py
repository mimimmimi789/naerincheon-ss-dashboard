"""휴대폰 알림센터로 SS 예측 요약 보내기 (웹 푸시). GitHub Actions(.github/workflows/notify.yml)가 실행한다.

흐름
1. 저장소 비밀값 PUSH_SECRET 하나에서 열쇠 2쌍과 우편함 이름을 만든다.
   - VAPID 열쇠: 푸시 서비스(구글·애플·삼성)에 "이 대시보드가 보낸 알림"임을 증명
   - 상자 열쇠: 휴대폰이 보낸 등록 정보를 GitHub Actions만 열어 볼 수 있게 암호화
   공개 열쇠와 우편함 이름은 docs/push_key.json으로 대시보드에 올라간다.
2. 휴대폰 대시보드에서 '알림 받기' → 그 자리에서 오늘 요약 알림 표시 + 등록 정보를 암호화해 ntfy.sh 우편함에 넣는다.
3. Actions가 10분마다(poll) 우편함 확인 → 새 기기를 암호화 목록(data/push/subs.enc)에 저장하고 '등록 완료' 알림.
4. PC가 매일 07:30 갱신을 올리면(docs/latest.json 변경) Actions가 모든 기기에 1·3일 요약을 보낸다.

사용
  python pipeline/notify.py poll     우편함 확인 (Actions가 10분마다 실행)
  python pipeline/notify.py send     지금 요약 알림 보내기
  python pipeline/notify.py test     시험 알림
  python pipeline/notify.py list     등록된 기기 수
"""
import base64, hashlib, hmac, json, os, sys, time
from pathlib import Path
from urllib.parse import urlparse

try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
PDIR = ROOT / 'data' / 'push'          # .gitignore: 비공개 열쇠·등록 기기 목록은 PC에만
SITE = 'https://mimimmimi789.github.io/naerincheon-ss-dashboard/'
RELAY = 'https://ntfy.sh/'
# 알림을 보낼 수 있는 푸시 서비스 주소만 허용 (구글 FCM = Chrome·삼성 인터넷, 애플, 모질라, 마이크로소프트)
PUSH_HOSTS = ('fcm.googleapis.com', 'web.push.apple.com', '.push.services.mozilla.com', '.notify.windows.com')
MAX_SUBS = 50
CHUNK = 4096


def b64e(b): return base64.urlsafe_b64encode(b).rstrip(b'=').decode()
def b64d(s): s = s.strip(); return base64.urlsafe_b64decode(s + '=' * (-len(s) % 4))
def pub_raw(k): return k.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
def pub_from_raw(b): return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b)
def hkdf_extract(salt, ikm): return hmac.new(salt, ikm, hashlib.sha256).digest()
def hkdf_expand(prk, info, n): return hmac.new(prk, info + b'\x01', hashlib.sha256).digest()[:n]


def _secret():
    v = os.environ.get('PUSH_SECRET', '').strip()
    if not v and (PDIR / 'secret.txt').exists(): v = (PDIR / 'secret.txt').read_text(encoding='utf-8').strip()   # PC에서 시험할 때
    if len(v) < 16: raise SystemExit('PUSH_SECRET이 없어요: GitHub 저장소 Settings → Secrets → Actions에 PUSH_SECRET을 넣어 주세요.')
    return v.encode()


def _derive(label, n=32): return hmac.new(_secret(), label.encode(), hashlib.sha256).digest()[:n]


def _ec(label):
    order = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    return ec.derive_private_key(int.from_bytes(_derive(label), 'big') % (order - 1) + 1, ec.SECP256R1())


def ensure_keys():
    """비밀값 하나(PUSH_SECRET)에서 열쇠 2쌍과 우편함 이름을 만들고, 대시보드용 공개 정보(docs/push_key.json)를 맞춰 둔다."""
    v, b = _ec('vapid'), _ec('box'); topic = 'naerincheon-ss-' + b64e(_derive('topic', 12))
    pub = {'vapid': b64e(pub_raw(v)), 'box': b64e(pub_raw(b)), 'relay': RELAY + topic}
    pk = DOCS / 'push_key.json'
    if not pk.exists() or json.loads(pk.read_text(encoding='utf-8')) != pub:
        pk.write_text(json.dumps(pub, indent=1), encoding='utf-8')
    return v, b, topic


# ---- 웹 푸시 암호화 (RFC 8291 aes128gcm) + VAPID (RFC 8292) ----
def encrypt(sub, payload: bytes, as_key=None, salt=None):
    ua_pub = b64d(sub['keys']['p256dh']); auth = b64d(sub['keys']['auth'])
    as_key = as_key or ec.generate_private_key(ec.SECP256R1()); as_pub = pub_raw(as_key)
    salt = salt or os.urandom(16)
    shared = as_key.exchange(ec.ECDH(), pub_from_raw(ua_pub))
    ikm = hkdf_expand(hkdf_extract(auth, shared), b'WebPush: info\x00' + ua_pub + as_pub, 32)
    prk = hkdf_extract(salt, ikm)
    cek = hkdf_expand(prk, b'Content-Encoding: aes128gcm\x00', 16)
    nonce = hkdf_expand(prk, b'Content-Encoding: nonce\x00', 12)
    ct = AESGCM(cek).encrypt(nonce, payload + b'\x02', None)
    return salt + CHUNK.to_bytes(4, 'big') + bytes([len(as_pub)]) + as_pub + ct


def vapid_header(vkey, endpoint):
    u = urlparse(endpoint)
    head = b64e(json.dumps({'typ': 'JWT', 'alg': 'ES256'}, separators=(',', ':')).encode())
    body = b64e(json.dumps({'aud': f'{u.scheme}://{u.netloc}', 'exp': int(time.time()) + 12 * 3600, 'sub': SITE}, separators=(',', ':')).encode())
    r, s = decode_dss_signature(vkey.sign(f'{head}.{body}'.encode(), ec.ECDSA(hashes.SHA256())))
    jwt = f'{head}.{body}.' + b64e(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))
    return f'vapid t={jwt}, k={b64e(pub_raw(vkey))}'


def host_ok(endpoint):
    u = urlparse(endpoint)
    return u.scheme == 'https' and any(u.hostname == h or (h.startswith('.') and (u.hostname or '').endswith(h)) for h in PUSH_HOSTS)


# ---- 등록 기기 ----
def subs_load():
    """등록 기기 목록은 비밀값으로 암호화해 저장소(data/push/subs.enc)에 둔다: 공개 저장소여도 남이 못 읽음."""
    p = PDIR / 'subs.enc'
    if not p.exists(): return []
    raw = b64d(p.read_text(encoding='utf-8'))
    return json.loads(AESGCM(_derive('subs-file', 16)).decrypt(raw[:12], raw[12:], None))


def subs_save(S):
    PDIR.mkdir(parents=True, exist_ok=True); n = os.urandom(12)
    (PDIR / 'subs.enc').write_text(b64e(n + AESGCM(_derive('subs-file', 16)).encrypt(n, json.dumps(S).encode(), None)) + '\n', encoding='utf-8')


def push_one(vkey, sub, msg, ttl=12 * 3600):
    data = encrypt(sub, json.dumps(msg, ensure_ascii=False).encode())
    r = requests.post(sub['endpoint'], data=data, timeout=30, headers={
        'Authorization': vapid_header(vkey, sub['endpoint']), 'Content-Encoding': 'aes128gcm',
        'Content-Type': 'application/octet-stream', 'TTL': str(ttl), 'Urgency': 'normal'})
    return r.status_code


def push_all(msg, only=None):
    vkey, _, _ = ensure_keys(); S = subs_load(); keep, ok = [], 0
    for s in S:
        if only is not None and s['endpoint'] not in only: keep.append(s); continue
        try: code = push_one(vkey, s, msg)
        except requests.RequestException as e: code = None; print('알림 전송 실패(네트워크):', type(e).__name__)
        if code in (404, 410): print('만료된 기기 삭제:', urlparse(s['endpoint']).hostname); continue   # 알림 끔·앱 삭제
        if code and code < 300: ok += 1
        elif code: print('알림 전송 실패', code, urlparse(s['endpoint']).hostname)
        keep.append(s)
    if len(keep) != len(S): subs_save(keep)
    return ok


# ---- 알림 문구 (대시보드 맨 위 한 줄 결론과 같은 값: 관측만 입력한 예측 중앙값) ----
LV = ['평상', '관심', '대응 검토']
TAIL = ['평소 범위 유지 전망', '평소보다 높아질 수 있어 관심', '25 mg/L 이상 가능, 대응 검토']


def fmt(v): return f'{v:,.1f}'


def summary(latest=None, p90=10.2):
    j = latest or json.loads((DOCS / 'latest.json').read_text(encoding='utf-8'))
    H = j['forecast']['h']; lv, parts = 0, []
    today = time.strftime('%Y-%m-%d', time.gmtime(time.time() + 9 * 3600))
    for h in ('1', '3'):
        z = H.get(h)
        if not z or not z.get('obs'): continue
        d = z['target']; v = z['obs'][0]
        lab = ('오늘 ' if d == today else '') + f'{int(d[5:7])}/{int(d[8:10])}'
        parts.append(f'{lab} {fmt(v)}' + (' mg/L' if not parts else ''))
        m = max([v] + ([z['kma']['v'][0]] if z.get('kma') and z['kma'].get('v') else []))
        if h == '1' and m >= 25: lv = 2
        elif m >= p90: lv = max(lv, 1)
    return {'title': f'내린천2 SS · {LV[lv]}', 'body': ' · '.join(parts) + f'\n{TAIL[lv]}',
            'tag': 'ss-daily', 'url': SITE, 'lv': lv}


# ---- 우편함(ntfy.sh) 확인: 휴대폰이 넣은 암호화 등록 정보 꺼내기 ----
def open_box(bkey, blob):
    """휴대폰(WebCrypto)이 ECDH P-256 + HKDF-SHA256 + AES-GCM으로 잠근 등록 정보 열기."""
    d = json.loads(blob)
    shared = bkey.exchange(ec.ECDH(), pub_from_raw(b64d(d['k'])))
    key = hkdf_expand(hkdf_extract(b'naerincheon-ss-box', shared), b'subscription', 32)
    return json.loads(AESGCM(key).decrypt(b64d(d['n']), b64d(d['c']), None))


def poll(quiet=False):
    vkey, bkey, topic = ensure_keys()
    st = PDIR / 'poll_since.txt'; since = st.read_text().strip() if st.exists() else '12h'
    try:
        r = requests.get(f'{RELAY}{topic}/json', params={'poll': '1', 'since': since}, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print('우편함 확인 실패:', type(e).__name__)
        return 0
    S = subs_load(); n0 = len(S); have = {s['endpoint'] for s in S}; new, last, off = [], None, 0
    for line in r.text.splitlines():
        try: m = json.loads(line)
        except ValueError: continue
        if m.get('event') != 'message': continue
        last = str(m.get('time') or '') or last     # 다음 확인은 이 시각부터 (같은 글이 다시 와도 기기 주소로 중복 걸러냄)
        try: sub = open_box(bkey, m.get('message', ''))
        except Exception: continue          # 다른 사람이 넣은 엉뚱한 글은 무시
        ep = sub.get('endpoint', '')
        if sub.get('op') == 'off':          # '알림 끄기'
            n1 = len(S); S = [s for s in S if s['endpoint'] != ep]; have.discard(ep); off += n1 - len(S); continue
        if not host_ok(ep) or not sub.get('keys', {}).get('p256dh') or not sub['keys'].get('auth') or ep in have: continue
        if len(S) >= MAX_SUBS: break
        S.append({'endpoint': ep, 'keys': {'p256dh': sub['keys']['p256dh'], 'auth': sub['keys']['auth']},
                  'device': str(sub.get('device', ''))[:40], 'added': time.strftime('%Y-%m-%d %H:%M')})
        have.add(ep); new.append(ep)
    if new or off: subs_save(S)
    if last: PDIR.mkdir(parents=True, exist_ok=True); st.write_text(last)
    if new:
        try: s = summary()
        except Exception: s = {'body': '', 'lv': 0}
        msg = {'title': '알림 등록 완료 · 내린천2 SS', 'body': (s['body'] + '\n' if s.get('body') else '') + '매일 아침 07:30 갱신 후 요약을 보내드려요.',
               'tag': 'ss-daily', 'url': SITE}
        n = push_all(msg, only=set(new))
        print(f'새 기기 {len(new)}대 등록, 확인 알림 {n}건 보냄')
    return len(new)


def daily(p90=10.2):
    """매일 갱신 뒤 Actions에서 호출: 우편함 확인 후 모든 기기에 요약 발송."""
    try:
        poll(quiet=True)
        if not subs_load(): return
        msg = summary(p90=p90); n = push_all(msg)
        print(f'알림 {n}건 보냄: {msg["title"]} | {msg["body"].splitlines()[0]}')
    except SystemExit: raise
    except Exception as e:
        print('알림 단계 오류:', type(e).__name__); raise


if __name__ == '__main__':
    cmd = (sys.argv[1:] or ['list'])[0]
    if cmd == 'poll': poll()
    elif cmd == 'send': daily()
    elif cmd == 'test': ensure_keys(); print('보냄', push_all({'title': '시험 알림 · 내린천2 SS', 'body': '알림이 잘 도착했어요.', 'tag': 'ss-test', 'url': SITE}), '건')
    elif cmd == 'list': ensure_keys(); S = subs_load(); print(f'등록 기기 {len(S)}대'); [print(' -', s.get('device') or '?', s.get('added'), urlparse(s['endpoint']).hostname) for s in S]
    else: print(__doc__)
