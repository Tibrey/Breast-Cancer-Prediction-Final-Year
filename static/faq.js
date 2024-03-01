document.addEventListener("DOMContentLoaded", function () {
    // Get all the buttons with class "accordion"
    var accordionButtons = document.querySelectorAll(".accordion");

    // Loop through each button
    accordionButtons.forEach(function (button) {
        // Add click event listener to each button
        button.addEventListener("click", function () {
            // Get the corresponding paragraph
            var paragraph = this.nextElementSibling;

            // Toggle the "active" class on the paragraph to show/hide it
            if (paragraph && paragraph.classList.contains("paragraph")) {
                paragraph.classList.toggle("active");
            }
        });
    });
});
