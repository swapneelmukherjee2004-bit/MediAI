const { chromium } = require('playwright');
const path = require('path');

(async () => {
  console.log('Launching browser...');
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1440, height: 900 });

  const artifactDir = '/Users/swapneelmukherjee/.gemini/antigravity/brain/83e08b21-8ef7-4a9e-a15a-9389c411785a';

  try {
    console.log('Navigating to homepage...');
    await page.goto('http://localhost:3000/');
    await page.waitForTimeout(3000); // Wait for animations/rendering
    await page.screenshot({ path: path.join(artifactDir, '01_homepage.png'), fullPage: true });
    console.log('Saved 01_homepage.png');

    console.log('Navigating to login page...');
    await page.goto('http://localhost:3000/login');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(artifactDir, '02_login.png'), fullPage: true });
    console.log('Saved 02_login.png');

    console.log('Navigating to diagnose page...');
    await page.goto('http://localhost:3000/diagnose');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(artifactDir, '03_diagnose.png'), fullPage: true });
    console.log('Saved 03_diagnose.png');
    
    console.log('Navigating to diseases directory...');
    await page.goto('http://localhost:3000/diseases');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(artifactDir, '04_diseases.png'), fullPage: true });
    console.log('Saved 04_diseases.png');
  } catch (err) {
    console.error('Error taking screenshots:', err);
  } finally {
    await browser.close();
    console.log('Screenshots finished!');
  }
})();
