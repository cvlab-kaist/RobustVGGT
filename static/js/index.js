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

  // Collect sections corresponding to nav items
  navItems.forEach(item => {
    const id = item.getAttribute('data-target');
    const section = document.getElementById(id);
    if (section) {
      sections.push({ id, element: section, navItem: item });
    }
  });

  // Flag to disable observer during manual scroll/click
  let isManualScroll = false;
  let scrollTimeout;

  // IntersectionObserver options
  // rootMargin: '-20% 0px -60% 0px' creates a horizontal band across the viewport
  // from 20% down to 40% down. A section is "intersecting" if it crosses this band.
  const observerOptions = {
    root: null,
    rootMargin: '-20% 0px -60% 0px',
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    if (isManualScroll) return;

    entries.forEach(entry => {
      if (entry.isIntersecting) {
        // Remove active class from all items
        navItems.forEach(item => item.classList.remove('active'));

        // Find the matching nav item and add active class
        const id = entry.target.getAttribute('id');
        const navItem = document.querySelector(`.nav-item[data-target="${id}"]`);
        if (navItem) {
          navItem.classList.add('active');
        }
      }
    });
  }, observerOptions);

  // Start observing all sections
  sections.forEach(section => {
    observer.observe(section.element);
  });

  // Add click event listeners for immediate feedback and locking
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      // Allow default anchor click behavior (scrolling)

      // Set manual scroll flag to ignore observer updates
      isManualScroll = true;

      // Update active class immediately
      navItems.forEach(nav => nav.classList.remove('active'));
      item.classList.add('active');

      // Clear existing timeout
      if (scrollTimeout) clearTimeout(scrollTimeout);

      // Reset flag after scroll animation finishes (approx 1000ms)
      scrollTimeout = setTimeout(() => {
        isManualScroll = false;
        // Optional: Re-check intersection if needed, but usually not necessary
        // as the user is now at the target section.
      }, 1000);
    });
  });
});
