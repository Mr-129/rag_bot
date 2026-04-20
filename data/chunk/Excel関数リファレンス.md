---
title: "Excelのよく使う関数リファレンス"
product: "Microsoft365"
topic: "Excel基本"
doc_type: "reference"
os: ["Windows", "macOS"]
tags: ["Excel", "関数", "VLOOKUP", "XLOOKUP", "IF", "SUMIFS", "COUNTIFS", "ピボットテーブル"]
source_path: "data/chunk/Excel関数リファレンス.md"
related_ids: []
created_at: "2026-03-22"
---
# Excelのよく使う関数リファレンス

## 検索・参照

### XLOOKUP（推奨）
VLOOKUPの後継。列の位置に依存しない検索が可能。
```
=XLOOKUP(検索値, 検索範囲, 戻り範囲, [見つからない場合])
```
例：社員番号から氏名を取得
```
=XLOOKUP(A2, 社員マスタ!A:A, 社員マスタ!B:B, "該当なし")
```

### VLOOKUP（従来方式）
```
=VLOOKUP(検索値, テーブル範囲, 列番号, FALSE)
```
※ 検索列はテーブルの左端に必要。新規作成にはXLOOKUP推奨。

## 条件付き集計

### SUMIFS
```
=SUMIFS(合計範囲, 条件範囲1, 条件1, [条件範囲2, 条件2]...)
```
例：営業部の4月売上合計
```
=SUMIFS(D:D, B:B, "営業部", C:C, ">=2026/04/01", C:C, "<2026/05/01")
```

### COUNTIFS
```
=COUNTIFS(条件範囲1, 条件1, [条件範囲2, 条件2]...)
```

## 条件分岐

### IF / IFS
```
=IF(条件, 真の場合, 偽の場合)
=IFS(条件1, 値1, 条件2, 値2, TRUE, デフォルト値)
```

## テキスト操作
| 関数 | 用途 | 例 |
|------|------|-----|
| LEFT | 左からN文字 | `=LEFT(A1, 3)` |
| RIGHT | 右からN文字 | `=RIGHT(A1, 4)` |
| MID | 途中からN文字 | `=MID(A1, 2, 3)` |
| CONCAT | 文字列結合 | `=CONCAT(A1, "-", B1)` |
| TEXT | 書式変換 | `=TEXT(A1, "yyyy/mm/dd")` |

## ピボットテーブル作成
1. 集計したいデータ範囲を選択
2. 「挿入」→「ピボットテーブル」
3. 行・列・値のフィールドをドラッグ＆ドロップ
4. 値の集計方法（合計/平均/個数）を選択
