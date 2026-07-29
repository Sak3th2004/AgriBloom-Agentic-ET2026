/** Quick-symptom chips: id maps to i18n key `symptom.<id>`; query is what we send to the API. */
export interface QuickSymptom {
  id: string;
  emoji: string;
  query: string;
}

export const QUICK_SYMPTOMS: QuickSymptom[] = [
  { id: "leaf_spots", emoji: "🟤", query: "My crop leaves have dark spots on them" },
  { id: "yellow_leaves", emoji: "🍂", query: "The leaves are turning yellow" },
  { id: "insects", emoji: "🐛", query: "There are insects attacking my crop" },
  { id: "white_fungus", emoji: "🍄", query: "White fungus powder on the leaves" },
  { id: "wilting", emoji: "🥀", query: "The plants are wilting and drooping" },
  { id: "healthy_check", emoji: "✅", query: "Is my crop healthy? General check" },
];
