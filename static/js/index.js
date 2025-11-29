window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function () { return false; };
  image.oncontextmenu = function () { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function () {
  // Check for click events on the navbar burger icon
  $(".navbar-burger").click(function () {
    // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");

  });

  var options = {
    slidesToScroll: 1,
    slidesToShow: 3,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000,
  }

  // Initialize all div with carousel class
  var carousels = bulmaCarousel.attach('.carousel', options);

  // Loop on each carousel initialized
  for (var i = 0; i < carousels.length; i++) {
    // Add listener to  event
    carousels[i].on('before:show', state => {
      console.log(state);
    });
  }

  // Access to bulmaCarousel instance of an element
  var element = document.querySelector('#my-element');
  if (element && element.bulmaCarousel) {
    // bulmaCarousel instance is available as element.bulmaCarousel
    element.bulmaCarousel.on('before-show', function (state) {
      console.log(state);
    });
  }

  /*var player = document.getElementById('interpolation-video');
  player.addEventListener('loadedmetadata', function() {
    $('#interpolation-slider').on('input', function(event) {
      console.log(this.value, player.duration);
      player.currentTime = player.duration / 100 * this.value;
    })
  }, false);*/
  preloadInterpolationImages();

  $('#interpolation-slider').on('input', function (event) {
    setInterpolationImage(this.value);
  });
  setInterpolationImage(0);
  $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

  bulmaSlider.attach();

})

document.addEventListener('DOMContentLoaded', () => {
  const navItems = document.querySelectorAll('.nav-item');
  const sections = [];

  // Collect all sections and their corresponding nav items
  navItems.forEach(item => {
    const targetId = item.getAttribute('data-target');
    const section = document.getElementById(targetId);
    if (section) {
      sections.push({
        id: targetId,
        element: section,
        navItem: item
      });
    }
  });

  if (sections.length === 0) return;

  // Track if user manually clicked (to prevent immediate override)
  let isUserClicking = false;
  let clickTimeout = null;

  // Function to update active nav item based on scroll position
  function updateActiveNavOnScroll() {
    if (isUserClicking) return; // Don't update while user is clicking

    const scrollPosition = window.scrollY + window.innerHeight / 3; // Check at 1/3 from top

    let currentSection = null;

    // Find which section we're currently in
    for (let i = sections.length - 1; i >= 0; i--) {
      const section = sections[i];
      const sectionTop = section.element.offsetTop;

      if (scrollPosition >= sectionTop) {
        currentSection = section;
        break;
      }
    }

    // If we found a current section, update the active state
    if (currentSection) {
      navItems.forEach(item => item.classList.remove('active'));
      currentSection.navItem.classList.add('active');
    }
  }

  // Listen to scroll events with throttling
  let scrollTimeout = null;
  window.addEventListener('scroll', () => {
    if (scrollTimeout) {
      clearTimeout(scrollTimeout);
    }

    scrollTimeout = setTimeout(() => {
      updateActiveNavOnScroll();
    }, 50); // Small delay for performance
  }, { passive: true });

  // Handle nav item clicks
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      // Immediately update active state
      navItems.forEach(nav => nav.classList.remove('active'));
      item.classList.add('active');

      // Set flag to prevent scroll listener from overriding
      isUserClicking = true;

      // Clear any existing timeout
      if (clickTimeout) {
        clearTimeout(clickTimeout);
      }

      // Reset flag after scroll animation completes
      clickTimeout = setTimeout(() => {
        isUserClicking = false;
        // Re-check position after scroll settles
        updateActiveNavOnScroll();
      }, 1000);
    });
  });

  // Initial update on page load
  updateActiveNavOnScroll();
});
