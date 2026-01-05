document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const btn = document.getElementById('predictBtn');
    const originalBtnText = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
    btn.disabled = true;

    // Collect result section
    const resultSection = document.getElementById('resultSection');
    resultSection.classList.remove('visible');
    resultSection.classList.remove('hidden');

    // Build JSON data
    const formData = new FormData(e.target);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = isNaN(value) ? value : Number(value);
    });

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        
        // Update UI
        setTimeout(() => {
            updateResult(result);
            resultSection.classList.add('visible');
            btn.innerHTML = originalBtnText;
            btn.disabled = false;
        }, 500); // Simulate processing time for effect

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during prediction. Please check the console.');
        btn.innerHTML = originalBtnText;
        btn.disabled = false;
    }
});

function updateResult(result) {
    const riskMeter = document.getElementById('riskMeter');
    const riskScore = document.getElementById('riskScore');
    const label = document.getElementById('predictionLabel');
    const levelText = document.getElementById('riskLevelText');
    const levelTag = document.getElementById('riskLevelTag');
    const timestamp = document.getElementById('timestamp');

    // Stats
    const confidence = parseFloat(result.confidence * 100).toFixed(1);
    const isDisease = result.prediction === 1;
    
    // Update text
    label.innerText = result.prediction_label;
    label.style.color = isDisease ? 'var(--danger)' : 'var(--success)';
    
    riskScore.innerText = `${confidence}%`;
    levelText.innerText = result.risk_level;
    timestamp.innerText = new Date(result.timestamp).toLocaleString();

    // Visuals
    let color = 'var(--success)';
    if (result.risk_level === 'Moderate') color = 'var(--warning)';
    if (result.risk_level === 'High' || result.risk_level === 'Very High') color = 'var(--danger)';

    levelTag.style.border = `1px solid ${color}`;
    levelTag.style.color = color;

    // Meter Animation
    // Map confidence (0-1) to rotation (-45 to 225 deg) = 270 deg span
    // Note: Confidence is probability of the Predicted Class
    // If prediction is 0 (No Disease), high confidence means LOW risk.
    // If prediction is 1 (Disease), high confidence means HIGH risk.

    let riskFactor = 0; // 0 to 1 scale for visual
    if (isDisease) {
        riskFactor = result.confidence;
    } else {
        // If not disease, risk is inverse of confidence in 'No Disease'
        // Actually, let's simplify. If pred=0, risk is low. If pred=1, risk is high.
        // We want the meter to fill up for risk.
        // If pred=0 with 90% conf, risk ~ 10%.
        riskFactor = 1 - result.confidence;
        if (riskFactor < 0) riskFactor = 0;
    }

    // However, the model returns confidence of the PREDICTED class.
    // So if Prediction=1 (Disease), Prob=0.8 -> Risk 0.8
    // If Prediction=0 (No Disease), Prob=0.8 -> Risk 0.2 (approx)
    
    // Let's rely on the risk_level logic from backend if possible, or calculate:
    // Backend logic:
    // disease_prob = probabilities[1]
    // if disease_prob < 0.3: Low
    // ...
    // The response doesn't give raw probability of class 1 directly if class is 0.
    // But we know pred = 1 means prob(1) >= 0.5 usually.
    // Let's just use the returned 'confidence' and 'prediction' to estimate fill.
    
    let fillPercentage = 0;
    if (result.prediction === 1) {
        fillPercentage = result.confidence;
    } else {
        fillPercentage = 0.2; // Visual baseline for low risk
    }

    // Meter rotation
    // -45deg is empty, 135deg is full (180 span for half circle?)
    // Our CSS defines it as border-top/right. 
    // Let's just create a full circle effect or simple color change.
    
    riskMeter.style.borderColor = color;
    // We can't easily animate stroke-dasharray on a div border, but we can rotate it.
    // Let's just change color for now, complex meter animation in vanilla CSS without SVG is tricky.
    // We already rotate it -45deg.
}
