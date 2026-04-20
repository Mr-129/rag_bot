---
title: "Bitlockerの回復キー取得方法"
product: "セキュリティ"
topic: "ディスク暗号化"
doc_type: "troubleshooting"
os: ["Windows"]
tags: ["Bitlocker", "暗号化", "回復キー", "ロック解除", "TPM", "ブルースクリーン", "AzureAD"]
source_path: "data/chunk/Bitlocker回復キー.md"
related_ids: []
created_at: "2026-04-01"
---
# Bitlockerの回復キー取得方法

## Bitlocker回復キーの入力を求められるケース
- BIOS/UEFIの設定を変更した
- ハードウェアの構成を変更した（メモリ増設など）
- Windows Updateの不具合
- TPM（セキュリティチップ）のリセット

## 回復キーの取得方法

### 方法1：Azure ADから取得（推奨）
1. 別のPCまたはスマートフォンでブラウザを開く
2. https://myaccount.microsoft.com にアクセス
3. 社内アカウントでサインイン
4. 「デバイス」→ 対象PCを選択
5. 「Bitlocker キーを表示」をクリック
6. 48桁の回復キーが表示される

### 方法2：IT部門に連絡
Azure ADから取得できない場合は、IT部門に以下を伝えて回復キーを発行してもらう：
- PC名（PCの底面ラベルに記載）
- シリアル番号
- 本人確認情報

## 回復キーの入力
1. Bitlocker回復画面でキーIDを確認
2. 取得した48桁の回復キーを入力
3. Windowsが正常に起動する
4. 起動後、IT部門に報告（再発防止の確認のため）

## 注意事項
- 回復キーは他人に共有しない
- 回復キーの入力が頻繁に求められる場合はTPMの故障の可能性あり → IT部門に確認
