# Vendored browser libraries

These are checked in rather than loaded from a CDN, unlike the Tailwind and
Cytoscape tags in `templates/index.html`.

The reason is specific to what they render. `chat-markdown.js` is the first
place the Hub moves from `textContent` to `innerHTML`, and the content it
renders is **model output** — which is partially influenceable by anything that
reaches Orion's context, including recall hits and fetched web content. A
compromised CDN serving a broken stylesheet and one serving script into the
surface that renders Orion's replies are not the same severity class.

Vendoring also means the sanitizer cannot fail open by failing to load. A
network hiccup that drops DOMPurify while `marked` is cached would otherwise
turn every reply into raw unsanitized HTML. `chat-markdown.js` refuses to render
markdown at all unless **both** libraries are present, falling back to
`textContent` — but not having a network in the path at all is the stronger
guarantee.

| file | version | source | sha256 |
|---|---|---|---|
| `marked-15.0.7.min.js` | 15.0.7 | `npm pack marked@15.0.7` → `package/marked.min.js` | `934e3e36e9e2da0afb1a6e75075bb0f09af05293a844e84a7477ef40911c349a` |
| `dompurify-3.2.4.min.js` | 3.2.4 | `npm pack dompurify@3.2.4` → `package/dist/purify.min.js` | `8eb41b658831fab175fad9bcd00fcb2d84e0ed3a25a55053d4ecd4444b8b43a0` |

Licenses are checked in beside each file (both MIT; DOMPurify is dual
MPL-2.0/Apache-2.0 with the full text in `dompurify-3.2.4.LICENSE`).

## Updating

```bash
npm pack marked@<version>
tar xzf marked-<version>.tgz
cp package/marked.min.js services/orion-hub/static/js/vendor/marked-<version>.min.js
```

Then update the filename in `templates/index.html`, the table above, and the
sha256. Keep the version in the filename — a cache-busting rename is the point,
and a bare `marked.min.js` makes it impossible to tell what is deployed.
