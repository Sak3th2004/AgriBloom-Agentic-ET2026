/** Quick-symptom chips: id maps to i18n key `symptom.<id>`; query is what we send to the API. */
export interface QuickSymptom {
  id: string;
  query: string;
}

export const QUICK_SYMPTOMS: QuickSymptom[] = [
  { id: "leaf_spots", query: "My crop leaves have dark spots on them" },
  { id: "yellow_leaves", query: "The leaves are turning yellow" },
  { id: "insects", query: "There are insects attacking my crop" },
  { id: "white_fungus", query: "White fungus powder on the leaves" },
  { id: "wilting", query: "The plants are wilting and drooping" },
  { id: "healthy_check", query: "Is my crop healthy? General check" },
];
