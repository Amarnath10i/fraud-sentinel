import { fmt } from "./format";

// Category codes are positions in this list (src/sentinel/features/rows.py).
export const CATEGORIES = [
  "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
  "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
  "shopping_net", "shopping_pos", "travel",
];

const LABELS: Record<string, string> = {
  amount: "Amount",
  hour: "Hour of day",
  day_of_week: "Day of week",
  is_night: "Night-time (22:00-04:00)",
  age_years: "Cardholder age",
  gender_m: "Gender",
  log_city_pop: "City size",
  dist_home_km: "Distance from home",
  category_code: "Merchant category",
  card__count_1h: "Card transactions, last hour",
  card__count_24h: "Card transactions, last 24h",
  card__count_7d: "Card transactions, last 7 days",
  card__sum_amount_1h: "Card spend, last hour",
  card__sum_amount_24h: "Card spend, last 24h",
  card__sum_amount_7d: "Card spend, last 7 days",
  card__max_amount_24h: "Largest amount, last 24h",
  card__max_amount_7d: "Largest amount, last 7 days",
  card__distinct_merchant_24h: "Distinct merchants, last 24h",
  card__distinct_category_24h: "Distinct categories, last 24h",
  card__since_last: "Time since previous transaction",
  card__count_all: "Card's lifetime transactions",
  card__mean_amount_all: "Card's lifetime average",
  card__std_amount_all: "Card's amount spread",
  card__since_first: "Card history length",
  card_category__count_all: "Card's history in this category",
  card_category__count_7d: "This category, last 7 days",
  card_category__mean_amount_all: "Card's average in this category",
  merchant__count_30d: "Merchant volume, 30 days",
  merchant__cb_count_30d: "Merchant chargebacks, 30 days",
  category__cb_count_30d: "Category chargebacks, 30 days",
  card__cb_count_all: "Card's reported frauds",
  card__cb_since_last: "Time since card's last chargeback",
  amount_to_card_mean: "Amount vs card average",
  amount_zscore_card: "Amount z-score for this card",
  amount_to_card_max_7d: "Amount vs card's 7-day max",
  amount_to_card_category_mean: "Amount vs card's category average",
  card_avg_amount_24h: "Card's average, last 24h",
  merchant_cb_rate_30d: "Merchant chargeback rate",
  category_cb_rate_30d: "Category chargeback rate",
};

export const featureLabel = (name: string) => LABELS[name] ?? name;

const MONEY = /^amount$|sum_amount|max_amount|mean_amount|avg_amount/;

export function featureValue(name: string, v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "none yet";
  if (name === "category_code") return CATEGORIES[v] ?? "unknown";
  if (name === "is_night") return v ? "yes" : "no";
  if (name === "gender_m") return v ? "M" : "F";
  if (name === "hour") return fmt.hour(v);
  if (name === "day_of_week") return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][v] ?? String(v);
  if (name === "dist_home_km") return Math.round(v) + " km";
  if (name === "age_years") return Math.round(v) + " years";
  if (name === "log_city_pop") return fmt.int(Math.exp(v)) + " people";
  if (name.includes("since")) return fmt.duration(v);
  if (name.includes("rate")) return (v * 100).toFixed(2) + "%";
  if (name === "amount_zscore_card") return v.toFixed(2);
  if (name.startsWith("amount_to")) return "×" + v.toFixed(2);
  if (MONEY.test(name)) return fmt.usd2(v);
  return Number.isInteger(v) ? fmt.int(v) : v.toFixed(2);
}
