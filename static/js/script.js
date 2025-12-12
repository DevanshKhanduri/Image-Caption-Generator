const form = document.getElementById("uploadForm");
const loader = document.getElementById("loader");
const btn = document.getElementById("submitBtn");

form.addEventListener("submit", function() {
    loader.style.display = "block";       // show loader
    btn.disabled = true;                  // disable button
    btn.innerText = "Generating...";      // change button text
});
