// Shared helpers for the blog.

function formatDate(iso) {
    const parts = iso.split("-");
    const months = ["January","February","March","April","May","June","July","August","September","October","November","December"];
    return months[parseInt(parts[1], 10) - 1] + " " + parseInt(parts[2], 10) + ", " + parts[0];
}

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
}

function stripFrontmatter(md) {
    if (md.startsWith("---")) {
        const end = md.indexOf("\n---", 3);
        if (end !== -1) {
            return md.slice(md.indexOf("\n", end + 4) + 1);
        }
    }
    return md;
}

// Replace math expressions with placeholders so marked.js doesn't touch them,
// then restore the math after markdown rendering. Placeholders use `@@` so
// marked won't interpret them as emphasis, links, or anything else.
function renderMarkdownWithMath(md) {
    const blocks = [];
    const stash = s => {
        blocks.push(s);
        return "@@MATH" + (blocks.length - 1) + "ENDMATH@@";
    };

    md = md.replace(/\$\$([\s\S]+?)\$\$/g, m => stash(m));
    md = md.replace(/(^|[^\\$])\$([^\n$]+?)\$/g, (_m, pre, inner) => pre + stash("$" + inner + "$"));

    let html = marked.parse(md);

    // Restore math, undoing markdown-style backslash escapes (\\\\ -> \\,
    // \_ -> _, etc.) so MathJax sees the same text Hexo's pipeline would.
    html = html.replace(/@@MATH(\d+)ENDMATH@@/g, (_m, i) => {
        return blocks[parseInt(i, 10)].replace(/\\([\\_*#{}\[\]()`])/g, "$1");
    });
    return html;
}

window.MathJax = {
    tex: {
        inlineMath: [["$", "$"], ["\\(", "\\)"]],
        displayMath: [["$$", "$$"], ["\\[", "\\]"]],
        processEscapes: true
    },
    options: {
        skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
    },
    startup: { typeset: false }
};
