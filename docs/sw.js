// 홈 화면(바탕화면)에 추가한 앱용: 항상 최신 화면을 먼저 받고, 인터넷이 안 될 때만 마지막으로 본 화면을 보여 줌
const C = 'ssdash-v1';   // 바꾸면 예전 저장본을 지움
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== C).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const r = e.request;
  if (r.method !== 'GET' || new URL(r.url).origin !== location.origin) return;
  e.respondWith(fetch(r).then(res => { if (res.ok) { const c = res.clone(); caches.open(C).then(x => x.put(r, c)); } return res; })
    .catch(() => caches.match(r).then(m => m || (r.mode === 'navigate' ? caches.match('./') : Response.error()))));
});

// 알림 (GitHub Actions의 pipeline/notify.py가 보냄): 알림센터에 표시, 누르면 대시보드 열기
self.addEventListener('push', e => {
  let m = {}; try { m = e.data ? e.data.json() : {}; } catch (x) { m = { body: e.data && e.data.text() }; }
  e.waitUntil(self.registration.showNotification(m.title || '내린천2 SS', {
    body: m.body || '', icon: 'icons/icon-192.png', badge: 'icons/badge-96.png', tag: m.tag || 'ss-daily', renotify: true,
    data: { url: m.url || './' } }));
});
self.addEventListener('notificationclick', e => {
  e.notification.close();
  const url = (e.notification.data && e.notification.data.url) || './';
  e.waitUntil(clients.matchAll({ type: 'window', includeUncontrolled: true }).then(ws => {
    for (const w of ws) if (w.url.startsWith(self.registration.scope) && 'focus' in w) return w.navigate(url).then(x => (x || w).focus()).catch(() => w.focus());
    return clients.openWindow(url);
  }));
});
