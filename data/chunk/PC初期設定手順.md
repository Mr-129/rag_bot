---
title: "PC初期設定手順書"
product: "社内PC"
topic: "初期設定"
doc_type: "howto"
os: ["Windows"]
tags: ["初期設定", "セットアップ", "新入社員", "キッティング", "Windows11", "Microsoftアカウント", "ドメイン参加", "プリンタ設定"]
source_path: "data/chunk/PC初期設定手順.md"
related_ids: []
created_at: "2026-01-20"
---
# 新規PC初期設定手順書

## 概要
新入社員や端末入替時のPC初期設定手順です。所要時間は約60分です。

## 前提条件
- Windows 11 Pro がプリインストールされていること
- 有線LANまたはWi-Fi環境があること
- Active Directoryドメインの管理者権限があること

## 設定手順

### 1. Windows初期セットアップ
1. PCの電源を入れる
2. 地域「日本」、キーボード「日本語」を選択
3. 「組織用に設定する」を選択
4. ドメイン名（corp.local）を入力してドメイン参加
5. ユーザーアカウント情報を入力

### 2. Windows Update
1. 設定 → Windows Update を開く
2. 「更新プログラムのチェック」をクリック
3. すべての更新が完了するまで繰り返す（再起動含む）
4. 完了まで30〜60分かかる場合がある

### 3. 必須ソフトウェアのインストール
以下のソフトウェアをSoftware Centerからインストール：
- Microsoft 365 Apps（Word, Excel, PowerPoint, Outlook, Teams）
- FortiClient VPN
- Adobe Acrobat Reader
- 7-Zip
- Chrome ブラウザ

### 4. プリンタの設定
1. 設定 → Bluetooth とデバイス → プリンターとスキャナー
2. 「プリンターまたはスキャナーの追加」をクリック
3. フロアに対応するプリンタを選択して追加

### 5. メール（Outlook）の設定
1. Outlookを起動
2. メールアドレスを入力して「接続」
3. 自動構成が完了するまで待機
4. 共有メールボックスが必要な場合は別途設定

### 6. 最終確認
- [ ] ドメインログインできる
- [ ] インターネットアクセスできる
- [ ] VPN接続できる
- [ ] メール送受信できる
- [ ] プリンタから印刷できる
- [ ] Teamsで通話テストできる
