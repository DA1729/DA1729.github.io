# Blog

Minimal hand-rolled static blog. No build step — just markdown files rendered in the browser.

## Layout

```
new_blog/
├── index.html       # homepage (lists posts from posts.json)
├── post.html        # post reader (reads ?post=<slug>)
├── tags.html        # posts grouped by tag
├── styles.css       # warm serif palette, same as portfolio
├── app.js           # shared helpers: markdown + math rendering
├── posts.json       # post index: slug, title, date, tags, description
├── posts/           # markdown sources, one file per post
└── images/          # images referenced by posts
```

## Writing a new post

1. Drop a markdown file into `posts/<Slug>.md`. Frontmatter is optional; if present, it's stripped before rendering.
2. Add an entry to `posts.json` with `slug` matching the filename (without `.md`).
3. Link images from the post with `images/foo.png` or the legacy `/images/foo.png` (both work).

## Math

Math is rendered with MathJax v3. Use `$...$` for inline and `$$...$$` for display.

The renderer replays markdown's backslash-escape rules inside math blocks so posts written for the Hexo pipeline (`\_`, `\\\\`, etc.) render identically.

## Serving

Because the site uses `fetch()`, it needs to be served over HTTP, not opened as a
`file://` URL:

```
python3 -m http.server 8000
# open http://localhost:8000/
```

For GitHub Pages or any static host, just push the whole directory.
