/**
 * Service Worker Registration for GPX Analyzer PWA
 */

(function() {
  'use strict';

  // Register service worker if supported
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', async () => {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js', {
          scope: '/'
        });

        console.log('[SW] Service worker registered:', registration.scope);

        // Handle updates
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing;
          console.log('[SW] New service worker installing...');

          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              // New version available
              console.log('[SW] New version available');
              showUpdateNotification();
            }
          });
        });
      } catch (error) {
        console.error('[SW] Service worker registration failed:', error);
      }
    });

    // Listen for version responses from SW
    navigator.serviceWorker.addEventListener('message', (event) => {
      if (event.data && event.data.version) {
        var el = document.querySelector('.footer-version');
        if (el) {
          el.textContent = el.textContent.replace(/· SW \S+/, '· SW ' + event.data.version);
        }
      }
    });

    // Handle controller change (new SW took over)
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      console.log('[SW] New service worker activated');
      updateVersionDisplay();
    });

    // Query SW version once it's ready
    navigator.serviceWorker.ready.then((reg) => updateVersionDisplay(reg.active));
  }

  /**
   * Query the active service worker for its version and update the footer
   */
  function updateVersionDisplay(sw) {
    sw = sw || navigator.serviceWorker.controller;
    if (!sw) return;
    sw.postMessage({ type: 'GET_VERSION' });
  }

  /**
   * Show notification that a new version is available
   */
  function showUpdateNotification() {
    // Create notification element if it doesn't exist
    let notification = document.getElementById('sw-update-notification');
    if (!notification) {
      notification = document.createElement('div');
      notification.id = 'sw-update-notification';
      notification.className = 'sw-update-notification';
      notification.innerHTML = `
        <span>A new version is available.</span>
        <button onclick="window.location.reload()">Refresh</button>
      `;
      document.body.appendChild(notification);
    }
  }
})();
