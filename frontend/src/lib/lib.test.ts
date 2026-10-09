import { describe, expect, it } from "vitest";
import { featureValue } from "./features";
import { fmt } from "./format";
import { parseInterval, parseTables } from "./tables";

describe("parseTables", () => {
  const md = [
    "# Report",
    "",
    "| model | PR-AUC | fit (s) |",
    "|:------|:-------|-------:|",
    "| LightGBM | 0.9903 (0.987-0.994) | 70 |",
    "| Rules | 0.2457 (0.218-0.270) | 0 |",
    "",
    "text in between",
    "",
    "| policy | total cost $ |",
    "|---|---|",
    "| approve everything | 1,133,325 (1,004,102-1,266,433) |",
  ].join("\n");

  it("finds every pipe table with its header and rows", () => {
    const tables = parseTables(md);
    expect(tables).toHaveLength(2);
    expect(tables[0].headers).toEqual(["model", "PR-AUC", "fit (s)"]);
    expect(tables[0].rows[1]).toEqual(["Rules", "0.2457 (0.218-0.270)", "0"]);
    expect(tables[1].rows).toHaveLength(1);
  });

  it("ignores lines that only look like tables", () => {
    expect(parseTables("| not a table |\njust text")).toEqual([]);
  });
});

describe("parseInterval", () => {
  it("reads a point estimate with its bootstrap interval", () => {
    expect(parseInterval("0.9903 (0.987-0.994)")).toEqual({ point: 0.9903, low: 0.987, high: 0.994 });
  });
  it("strips thousands separators and dollar signs", () => {
    expect(parseInterval("$1,133,325 (1,004,102-1,266,433)")).toEqual({ point: 1133325, low: 1004102, high: 1266433 });
  });
  it("handles plain numbers and rejects text", () => {
    expect(parseInterval("70")).toEqual({ point: 70 });
    expect(parseInterval("n/a")).toBeNull();
  });
});

describe("fmt.prob", () => {
  it("keeps tiny and near-certain probabilities readable", () => {
    expect(fmt.prob(0.00001)).toBe("<0.01%");
    expect(fmt.prob(0.0339)).toBe("3.4%");
    expect(fmt.prob(0.0012)).toBe("0.12%");
    expect(fmt.prob(0.5)).toBe("50%");
    expect(fmt.prob(0.999)).toBe(">99%");
  });
  it("compacts large dollar amounts", () => {
    expect(fmt.compactUsd(1_133_325)).toBe("$1.13M");
    expect(fmt.compactUsd(31_860)).toBe("$31.9K");
    expect(fmt.compactUsd(7_145)).toBe("$7,145");
  });
});

describe("featureValue", () => {
  it("turns model inputs back into something a reviewer understands", () => {
    expect(featureValue("category_code", 11)).toBe("shopping_net");
    expect(featureValue("hour", 2)).toBe("02:00");
    expect(featureValue("is_night", 1)).toBe("yes");
    expect(featureValue("card__since_last", 45_720)).toBe("12.7 h");
    expect(featureValue("amount", 1100.78)).toBe("$1,100.78");
    expect(featureValue("amount_to_card_category_mean", 18.93)).toBe("×18.93");
    expect(featureValue("merchant_cb_rate_30d", 0.0123)).toBe("1.23%");
    expect(featureValue("card__cb_since_last", null)).toBe("none yet");
  });
});
