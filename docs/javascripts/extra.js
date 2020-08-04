// The below can be used to open all nav links in the documentation, code found at
// from https://github.com/squidfunk/mkdocs-material/issues/767#issuecomment-384558269
// from the user Akkadius.
/*
document.addEventListener("DOMContentLoaded", function() {
    load_navpane();
});

function load_navpane() {
    var width = window.innerWidth;
    if (width <= 1200) {
        return;
    }

    var nav = document.getElementsByClassName("md-nav");
    for (var i = 0; i < nav.length; i++) {
        if (typeof nav.item(i).style === "undefined") {
            continue;
        }

        if (nav.item(i).getAttribute("data-md-level") && nav.item(i).getAttribute("data-md-component")) {
            nav.item(i).style.display = 'block';
            nav.item(i).style.overflow = 'visible';
        }
    }

    var nav = document.getElementsByClassName("md-nav__toggle");
    for(var i = 0; i < nav.length; i++) {
       nav.item(i).checked = true;
    }
}
*/