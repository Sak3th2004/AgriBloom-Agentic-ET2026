import { test, expect } from "@playwright/test";

/**
 * Happy-path smoke test (FRONTEND_PLAN.md §B2.5):
 * home → describe problem → submit (mock API) → result renders
 * the diagnosis and the compliance/safety card.
 */
test("farmer asks a question and gets a safe advisory", async ({ page }) => {
  await page.goto("/");

  await expect(
    page.getByRole("heading", { name: "AgriBloom" })
  ).toBeVisible();

  await page
    .getByRole("textbox", { name: "Or describe the problem" })
    .fill("Black spots spreading on my grape leaves");

  await page.getByRole("button", { name: "Get advice" }).click();

  // Mock diagnose takes a few seconds (simulated agent pipeline).
  await page.waitForURL("**/result/**", { timeout: 30_000 });

  // Diagnosis card
  await expect(page.getByText("Grape Downy Mildew").first()).toBeVisible();
  await expect(page.getByText("Confidence").first()).toBeVisible();

  // Compliance / safety card (must be loud and present)
  await expect(page.getByText("Safe to use").first()).toBeVisible();

  // Treatment text from the mock fixture
  await expect(page.getByText(/Bordeaux mixture/i).first()).toBeVisible();
});
