/**
 * ラッコキーワード抽出ツール（単体で動作、他ツールへの組み込みは前提にしない）
 *
 * ラッコキーワード (https://related-keywords.com/) には自動取得用の公式APIが無く、
 * 利用規約上スクレイピングも認められていないため、「ラッコキーワードの画面から
 * キーワードをコピーして貼り付け → このスクリプトが整形してキーワードリストに
 * 追加する」という半自動の抽出にしている。
 *
 * 使い方:
 *   1. ラッコキーワードで検索し、「全キーワードコピー」または結果を選択してコピー
 *   2. このスプレッドシートの「取込用」シートのA列に貼り付け（1行1キーワード、
 *      カンマ区切りでも改行区切りでも可）
 *   3. メニュー「ラッコキーワード取込」→「取込用シートから追加」を実行
 *   4. 「キーワードリスト」シートに重複を除いて追加される（ステータス=未使用）
 *
 * 抽出したキーワードは必要な分だけ他のツールへ手動でコピーして使う。
 */

const IMPORT_SHEET_NAME = '取込用';
const KEYWORD_SHEET_NAME = 'キーワードリスト';

const KEYWORD_HEADER = ['キーワード', 'ステータス', '取込日'];
const STATUS_UNUSED = '未使用';

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('ラッコキーワード取込')
    .addItem('取込用シートから追加', 'importKeywordsFromStagingSheet')
    .addToUi();
}

function importKeywordsFromStagingSheet() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const stagingSheet = getOrCreateSheet_(ss, IMPORT_SHEET_NAME);
  const keywordSheet = getOrCreateKeywordSheet_(ss);

  const rawValues = stagingSheet.getDataRange().getValues();
  const candidates = extractKeywords_(rawValues);

  if (candidates.length === 0) {
    SpreadsheetApp.getUi().alert('取込用シートにキーワードが見つかりませんでした。');
    return;
  }

  const existing = getExistingKeywords_(keywordSheet);
  const newKeywords = candidates.filter(function (kw) {
    return !existing.has(kw);
  });

  if (newKeywords.length > 0) {
    const today = new Date();
    const rows = newKeywords.map(function (kw) {
      return [kw, STATUS_UNUSED, today];
    });
    keywordSheet
      .getRange(keywordSheet.getLastRow() + 1, 1, rows.length, KEYWORD_HEADER.length)
      .setValues(rows);
  }

  stagingSheet.clearContents();

  const skipped = candidates.length - newKeywords.length;
  SpreadsheetApp.getUi().alert(
    newKeywords.length + '件のキーワードを追加しました（重複スキップ：' + skipped + '件）。'
  );
}

function extractKeywords_(rawValues) {
  const seen = new Set();
  const result = [];

  rawValues.forEach(function (row) {
    row.forEach(function (cell) {
      if (typeof cell !== 'string') return;
      cell
        .split(/[\n,、]+/)
        .map(function (s) {
          return s.trim();
        })
        .filter(function (s) {
          return s.length > 0;
        })
        .forEach(function (kw) {
          if (!seen.has(kw)) {
            seen.add(kw);
            result.push(kw);
          }
        });
    });
  });

  return result;
}

function getExistingKeywords_(keywordSheet) {
  const lastRow = keywordSheet.getLastRow();
  const existing = new Set();
  if (lastRow < 2) return existing;

  const values = keywordSheet.getRange(2, 1, lastRow - 1, 1).getValues();
  values.forEach(function (row) {
    const kw = String(row[0]).trim();
    if (kw.length > 0) existing.add(kw);
  });
  return existing;
}

function getOrCreateSheet_(ss, name) {
  let sheet = ss.getSheetByName(name);
  if (!sheet) {
    sheet = ss.insertSheet(name);
  }
  return sheet;
}

function getOrCreateKeywordSheet_(ss) {
  const sheet = getOrCreateSheet_(ss, KEYWORD_SHEET_NAME);
  if (sheet.getLastRow() === 0) {
    sheet.getRange(1, 1, 1, KEYWORD_HEADER.length).setValues([KEYWORD_HEADER]);
  }
  return sheet;
}
