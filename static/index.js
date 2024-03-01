document.addEventListener("DOMContentLoaded", function () {
  var menuToggle = document.getElementById("mobile-menu");
  var list = document.querySelector(".list");

  // Toggle the responsive menu
  menuToggle.addEventListener("click", function () {
    list.classList.toggle("show");
    menuToggle.classList.toggle("show");
  });

  // Close the menu when a menu item is clicked
  list.querySelectorAll("a").forEach(function (item) {
    item.addEventListener("click", function () {
      list.classList.remove(".show");
      menuToggle.classList.remove(".show");
    });
  });
});
