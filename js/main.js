// Portfolio Blog Theme JavaScript - Enhanced UX

document.addEventListener('DOMContentLoaded', function() {
  initDarkMode();
  initReadingProgress();
  initBackToTop();
  initSmoothScrolling();
  initIntersectionObserver();
});

function initDarkMode() {
  // Create dark mode toggle button
  const toggleButton = document.createElement('button');
  toggleButton.className = 'dark-mode-toggle';
  toggleButton.innerHTML = '☀️';
  toggleButton.setAttribute('aria-label', 'Toggle dark mode');
  
  // Insert toggle button into the page
  document.body.appendChild(toggleButton);
  
  // Check for saved theme preference or default to light mode
  const currentTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  if (currentTheme === 'dark' || (!currentTheme && prefersDark)) {
    document.documentElement.classList.add('dark');
    toggleButton.innerHTML = '🌙';
  }
  
  // Toggle dark mode
  toggleButton.addEventListener('click', function() {
    const isDark = document.documentElement.classList.contains('dark');
    
    if (isDark) {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
      toggleButton.innerHTML = '☀️';
    } else {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
      toggleButton.innerHTML = '🌙';
    }
  });
  
  // Listen for system theme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
    if (!localStorage.getItem('theme')) {
      if (e.matches) {
        document.documentElement.classList.add('dark');
        toggleButton.innerHTML = '🌙';
      } else {
        document.documentElement.classList.remove('dark');
        toggleButton.innerHTML = '☀️';
      }
    }
  });
}

function initReadingProgress() {
  // Create reading progress bar
  const progressBar = document.createElement('div');
  progressBar.className = 'reading-progress';
  document.body.appendChild(progressBar);
  
  // Update progress on scroll
  window.addEventListener('scroll', function() {
    const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;
    
    progressBar.style.width = scrolled + '%';
    
    // Show/hide progress bar
    if (winScroll > 100) {
      progressBar.classList.add('visible');
    } else {
      progressBar.classList.remove('visible');
    }
  });
}

function initBackToTop() {
  // Create back to top button
  const backToTop = document.createElement('div');
  backToTop.className = 'back-to-top';
  backToTop.setAttribute('aria-label', 'Back to top');
  document.body.appendChild(backToTop);
  
  // Show/hide back to top button
  window.addEventListener('scroll', function() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    
    if (scrollTop > 300) {
      backToTop.classList.add('visible');
    } else {
      backToTop.classList.remove('visible');
    }
  });
  
  // Scroll to top on click
  backToTop.addEventListener('click', function() {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });
}

function initSmoothScrolling() {
  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
}

function initIntersectionObserver() {
  // Subtle animation for elements coming into view
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  });
  
  // Observe elements for fade-in animation
  const elementsToObserve = document.querySelectorAll('.post-content p, .post-content h1, .post-content h2, .post-content h3, .post-content h4, .post-content h5, .post-content h6, .post-content blockquote, .post-content pre');
  
  elementsToObserve.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(15px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });
}