/**
 * Client-side router for offline PWA
 *
 * Intercepts navigation and fetches pages via JavaScript, swapping content
 * without full page reload. Works seamlessly online and offline via service worker cache.
 */
(function(global) {
    'use strict';

    const Router = {
        contentSelector: '#app-content',
        initialized: false,

        init() {
            if (this.initialized) return;
            this.initialized = true;

            // Intercept all internal link clicks
            document.addEventListener('click', this.handleClick.bind(this));

            // Handle browser back/forward
            window.addEventListener('popstate', this.handlePopState.bind(this));

            console.log('[Router] Initialized');
        },

        handleClick(e) {
            // Find the closest anchor element
            const link = e.target.closest('a[href]');
            if (!link) return;

            const url = new URL(link.href, location.origin);

            // Only handle same-origin links
            if (url.origin !== location.origin) return;

            // Don't intercept hash-only links (href="#" or href="#something")
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) return;

            // Don't intercept links that open in new tab
            if (link.target === '_blank') return;

            // Allow opting out with data attribute
            if (link.hasAttribute('data-no-router')) return;

            // Don't intercept links with download attribute
            if (link.hasAttribute('download')) return;

            // Don't intercept links with inline onclick handlers (modals, zoom controls, etc.)
            if (link.hasAttribute('onclick')) return;

            e.preventDefault();
            this.navigate(url.pathname + url.search + url.hash);
        },

        async navigate(url, pushState = true) {
            console.log('[Router] Navigating to:', url);

            try {
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const html = await response.text();
                this.updateContent(html, url, pushState);
            } catch (err) {
                console.error('[Router] Navigation failed:', err);
                // Fallback to traditional navigation
                location.href = url;
            }
        },

        updateContent(html, url, pushState) {
            // Parse the fetched HTML
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');

            // Extract new content
            const newContent = doc.querySelector(this.contentSelector);
            const currentContent = document.querySelector(this.contentSelector);

            if (!newContent || !currentContent) {
                console.warn('[Router] Content container not found, falling back');
                location.href = url;
                return;
            }

            // Update document title
            document.title = doc.title;

            // Check if body has page-specific classes and update
            const newBody = doc.body;
            const oldClasses = document.body.className;
            document.body.className = newBody.className;

            // Swap content
            currentContent.innerHTML = newContent.innerHTML;

            // Execute scripts in the new content
            this.executeScripts(currentContent);

            // Update URL in browser
            if (pushState) {
                history.pushState({ routerNav: true }, '', url);
            }

            // Scroll to top (or to hash if present)
            const hash = url.split('#')[1];
            if (hash) {
                const target = document.getElementById(hash);
                if (target) {
                    target.scrollIntoView();
                } else {
                    window.scrollTo(0, 0);
                }
            } else {
                window.scrollTo(0, 0);
            }

            // Dispatch a custom event for any listeners
            window.dispatchEvent(new CustomEvent('router:navigate', { detail: { url } }));
        },

        executeScripts(container) {
            // Find all script elements and re-execute them
            const scripts = container.querySelectorAll('script');
            scripts.forEach(oldScript => {
                const newScript = document.createElement('script');

                // Copy attributes
                Array.from(oldScript.attributes).forEach(attr => {
                    newScript.setAttribute(attr.name, attr.value);
                });

                if (oldScript.src) {
                    // External script - will be loaded by browser
                    newScript.src = oldScript.src;
                } else {
                    // Inline script - copy the content
                    newScript.textContent = oldScript.textContent;
                }

                // Replace old script with new one (triggers execution)
                oldScript.parentNode.replaceChild(newScript, oldScript);
            });
        },

        handlePopState(event) {
            // Browser back/forward button pressed
            console.log('[Router] Popstate:', location.pathname);
            this.navigate(location.pathname + location.search + location.hash, false);
        }
    };

    // Auto-init when DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => Router.init());
    } else {
        Router.init();
    }

    // Expose Router globally
    global.Router = Router;

})(window);
