// Run with: RF_MCP_BROWSER_BASE_URL=http://127.0.0.1:8765 npx playwright test
const {test, expect} = require('@playwright/test');

test('Listen live recovers and repeated stop/start ignores stale status', async ({page}) => {
  let statusCalls = 0;
  await page.route('**/api/live-audio/status', async route => {
    statusCalls++;
    await new Promise(resolve => setTimeout(resolve, statusCalls === 1 ? 700 : 5));
    await route.fulfill({json: {sessions: [{state: 'streaming', first_iq_monotonic: 1,
      first_encoded_chunk_monotonic: 2}]}});
  });
  await page.goto(process.env.RF_MCP_BROWSER_BASE_URL || 'http://127.0.0.1:8765/dashboard');
  await page.evaluate(() => { HTMLMediaElement.prototype.load = () => {};
    HTMLMediaElement.prototype.play = async function () { this.dispatchEvent(new Event('playing')); };
    HTMLMediaElement.prototype.pause = () => {}; });
  for (let cycle = 0; cycle < 3; cycle++) {
    const started = Date.now();
    await page.locator('#audioLiveButton').click();
    await expect(page.locator('#audioStatus')).toContainText('playing in');
    expect(Date.now() - started).toBeLessThan(1000);
    await page.locator('#audioStopButton').click();
    await expect(page.locator('#audioStatus')).toHaveText('Stopped');
  }
  await page.waitForTimeout(800);
  await expect(page.locator('#audioStatus')).toHaveText('Stopped');
});

test('waterfall renders a mocked row, reports errors, and restarts', async ({page}) => {
  let attempt = 0;
  await page.route('**/api/live-waterfall?*', route => {
    attempt++;
    const frame = attempt === 1
      ? {type:'error', error:'scripted disconnect'}
      : {type:'row', session_id:'fake', bits:8, row:'AA==', frequency_start_hz:1,
         frequency_end_hz:2};
    route.fulfill({contentType:'application/x-ndjson', body:JSON.stringify(frame)+'\n'});
  });
  await page.goto(process.env.RF_MCP_BROWSER_BASE_URL || 'http://127.0.0.1:8765/dashboard');
  await page.locator('#waterfallStart').click();
  await expect(page.locator('#waterfallStatus')).toContainText('failed');
  const started = Date.now();
  await page.locator('#waterfallStart').click();
  await expect(page.locator('#waterfallStatus')).toContainText(/Live|ended/);
  expect(Date.now() - started).toBeLessThan(1000);
});
