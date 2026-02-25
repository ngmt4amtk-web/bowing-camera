/**
 * bow-metrics.js — ボウイング評価アルゴリズム（純粋関数）
 */

const BowMetrics = (() => {
  // ── ユーティリティ ──

  function distance(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function angleBetween3(a, b, c) {
    // b を頂点として a-b-c の角度（度）
    const ba = { x: a.x - b.x, y: a.y - b.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };
    const dot = ba.x * bc.x + ba.y * bc.y;
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
    if (magBA === 0 || magBC === 0) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot / (magBA * magBC)))) * 180 / Math.PI;
  }

  function ema(prev, curr, alpha) {
    if (prev === null || prev === undefined) return curr;
    return prev + alpha * (curr - prev);
  }

  // ── M1: 弓の直線性 ──

  function computeBowStraightness(wristTrail) {
    // wristTrail: [{x, y, t}, ...] 直近フレーム
    if (wristTrail.length < 10) return { score: null, status: 'good' };

    // ストローク区切り: x方向の移動が反転したら新しいストローク
    const strokes = [];
    let currentStroke = [wristTrail[0]];
    let prevDx = 0;

    for (let i = 1; i < wristTrail.length; i++) {
      const dx = wristTrail[i].x - wristTrail[i - 1].x;
      // 移動量が微小なら無視（ノイズ除去）
      if (Math.abs(dx) < 0.002) {
        currentStroke.push(wristTrail[i]);
        continue;
      }
      // 方向反転の検出
      if (prevDx !== 0 && Math.sign(dx) !== Math.sign(prevDx)) {
        if (currentStroke.length >= 8) {
          strokes.push(currentStroke);
        }
        currentStroke = [wristTrail[i]];
      } else {
        currentStroke.push(wristTrail[i]);
      }
      prevDx = dx;
    }
    if (currentStroke.length >= 8) {
      strokes.push(currentStroke);
    }

    if (strokes.length === 0) return { score: null, status: 'good' };

    // 直近のストロークで評価
    const stroke = strokes[strokes.length - 1];
    const n = stroke.length;

    // 最小二乗法で直線フィット
    let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
    for (const p of stroke) {
      sumX += p.x;
      sumY += p.y;
      sumXX += p.x * p.x;
      sumXY += p.x * p.y;
    }
    const denom = n * sumXX - sumX * sumX;

    let rmse;
    if (Math.abs(denom) < 1e-10) {
      // ほぼ垂直なストローク → x方向のRMSE
      const meanX = sumX / n;
      let sse = 0;
      for (const p of stroke) {
        const d = p.x - meanX;
        sse += d * d;
      }
      rmse = Math.sqrt(sse / n);
    } else {
      const slope = (n * sumXY - sumX * sumY) / denom;
      const intercept = (sumY - slope * sumX) / n;
      // 各点から直線への距離のRMSE
      let sse = 0;
      for (const p of stroke) {
        const predicted = slope * p.x + intercept;
        const d = p.y - predicted;
        sse += d * d;
      }
      rmse = Math.sqrt(sse / n);
    }

    // ストローク長
    const first = stroke[0];
    const last = stroke[n - 1];
    const strokeLen = distance(first, last);
    if (strokeLen < 0.03) return { score: null, status: 'good' };

    // 正規化曲率スコア（%）
    const curvature = (rmse / strokeLen) * 100;
    // パーセント表示: 低い方が良い → 100 - curvature*5 でスコア化
    const score = Math.max(0, Math.min(100, 100 - curvature * 5));

    let status = 'good';
    if (curvature > 5) status = 'bad';
    else if (curvature > 2) status = 'warn';

    return { score: Math.round(score), curvature, status };
  }

  // ── M2: 弓の配分 ──

  function computeBowZone(shoulder, elbow, wrist) {
    // 腕全体の長さ（上腕+前腕）
    const upperArm = distance(shoulder, elbow);
    const forearm = distance(elbow, wrist);
    const armLength = upperArm + forearm;
    if (armLength < 0.01) return { zone: 'middle', extensionRatio: 0.5 };

    // 手首と肩の直線距離 / 腕全長 = 腕の伸び具合
    const wristShoulderDist = distance(shoulder, wrist);
    const extensionRatio = wristShoulderDist / armLength;

    // 伸び具合で判定（0=完全に畳む=元弓、1=完全に伸ばす=先弓）
    let zone;
    if (extensionRatio < 0.55) zone = 'frog';       // 元弓（腕が畳まれている）
    else if (extensionRatio < 0.78) zone = 'middle'; // 中弓
    else zone = 'tip';                                // 先弓（腕が伸びている）

    return { zone, extensionRatio: Math.round(extensionRatio * 100) / 100 };
  }

  function computeDistributionPercent(dist) {
    const total = dist.tip + dist.middle + dist.frog;
    if (total === 0) return { tip: 33, middle: 34, frog: 33, label: '--' };

    const tipPct = Math.round(dist.tip / total * 100);
    const frogPct = Math.round(dist.frog / total * 100);
    const midPct = 100 - tipPct - frogPct;

    // ラベル: 最も多いゾーン
    let label = '均等';
    const max = Math.max(tipPct, midPct, frogPct);
    if (max >= 50) {
      if (tipPct === max) label = '先弓寄り';
      else if (frogPct === max) label = '元弓寄り';
      else label = '中弓中心';
    } else if (max - Math.min(tipPct, midPct, frogPct) < 15) {
      label = '全弓';
    }

    return { tip: tipPct, middle: midPct, frog: frogPct, label };
  }

  // ── M3: 肘の高さ ──

  function computeElbowHeight(shoulder, elbow, rHip, lHip) {
    // 肩幅の基準: 右肩と左腰の距離の代わりに右肩-右腰を使う
    // → rHipがない場合は肩幅ベース不可なので簡易判定
    const shoulderElbowDy = elbow.y - shoulder.y;  // 正=肘が下、負=肘が上

    // 肩幅（体のスケール基準）
    let bodyScale;
    if (rHip) {
      bodyScale = Math.abs(rHip.y - shoulder.y);  // 肩-腰の距離
    } else {
      bodyScale = 0.3;  // フォールバック
    }
    if (bodyScale < 0.01) bodyScale = 0.3;

    const relativeHeight = shoulderElbowDy / bodyScale;

    // 理想範囲: -0.15 ~ 0.35（弦により変動するため広めに取る）
    let status = 'good';
    let label;
    if (relativeHeight < -0.15) {
      status = 'warn';
      label = '高すぎ';
    } else if (relativeHeight > 0.5) {
      status = 'bad';
      label = '低すぎ';
    } else if (relativeHeight > 0.35) {
      status = 'warn';
      label = 'やや低い';
    } else {
      label = 'OK';
    }

    return { relativeHeight: Math.round(relativeHeight * 100) / 100, status, label };
  }

  // ── M4: 肩の緊張 ──

  function computeShoulderTension(rShoulder, rEar, lShoulder, lEar, baseDist) {
    const rDist = distance(rShoulder, rEar);
    const lDist = distance(lShoulder, lEar);
    const avgDist = (rDist + lDist) / 2;

    // キャリブレーション不要版: 左右比を見る + baseDist比較
    let status = 'good';
    let label = 'リラックス';
    let tension = 0;

    if (baseDist !== null && baseDist > 0) {
      // baseDist からの変化率
      const change = (baseDist - avgDist) / baseDist;  // 正=肩が上がった
      tension = Math.max(0, Math.round(change * 100));

      if (change > 0.25) {
        status = 'bad';
        label = '力んでいます';
      } else if (change > 0.15) {
        status = 'warn';
        label = '少し力み';
      }
    }

    // 左右非対称チェック
    const asymmetry = Math.abs(rDist - lDist) / Math.max(rDist, lDist);
    if (asymmetry > 0.2) {
      if (status === 'good') {
        status = 'warn';
        label = '左右差あり';
      }
    }

    return { tension, status, label, currentDist: avgDist };
  }

  // ── アドバイス生成 ──

  function generateAdvice(metrics) {
    const issues = [];

    if (metrics.straightness && metrics.straightness.status === 'bad') {
      issues.push('弓が曲がっています。弓先まで弦と直角を保つよう意識してみてください');
    } else if (metrics.straightness && metrics.straightness.status === 'warn') {
      issues.push('弓がやや曲がっています。手首の柔軟性を意識してみましょう');
    }

    if (metrics.elbow && metrics.elbow.status === 'bad') {
      issues.push('肘が下がりすぎです。弦の高さに合わせて肘を上げましょう');
    } else if (metrics.elbow && metrics.elbow.label === '高すぎ') {
      issues.push('肘が高すぎます。力まず自然な高さに下ろしましょう');
    } else if (metrics.elbow && metrics.elbow.status === 'warn') {
      issues.push('肘をもう少し上げてみましょう');
    }

    if (metrics.shoulder && metrics.shoulder.status === 'bad') {
      issues.push('肩に力が入っています！一度息を吐いて肩を落としましょう');
    } else if (metrics.shoulder && metrics.shoulder.status === 'warn') {
      if (metrics.shoulder.label === '左右差あり') {
        issues.push('肩の高さに左右差があります。鏡で確認してみましょう');
      } else {
        issues.push('肩が少し上がっています。リラックスを意識してください');
      }
    }

    if (issues.length === 0) {
      if (metrics.distribution && metrics.distribution.label === '全弓') {
        return 'いい感じ！全弓をバランスよく使えています';
      }
      return 'フォームは良好です。この調子で続けましょう';
    }

    return issues[0];  // 最も重要な1つを表示
  }

  return {
    distance,
    angleBetween3,
    ema,
    computeBowStraightness,
    computeBowZone,
    computeDistributionPercent,
    computeElbowHeight,
    computeShoulderTension,
    generateAdvice,
  };
})();
