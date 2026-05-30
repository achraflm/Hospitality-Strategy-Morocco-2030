const puppeteer = require('puppeteer');

(async () => {
  try {
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
    
    // Wait a bit to ensure everything renders
    await new Promise(r => setTimeout(r, 2000));
    
    // Check if the sidebar exists
    const hasSidebar = await page.evaluate(() => {
      return !!document.querySelector('aside');
    });
    console.log('HAS SIDEBAR:', hasSidebar);
    
    await browser.close();
  } catch (err) {
    console.error('SCRIPT ERROR:', err);
  }
})();
