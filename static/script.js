document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Show loading state
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalHTML = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    submitBtn.disabled = true;
    
    try {
        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error making prediction: ' + error.message);
    } finally {
        // Reset button
        submitBtn.innerHTML = originalHTML;
        submitBtn.disabled = false;
    }
});

function displayResults(result) {
    const resultDiv = document.getElementById('result');
    const resultHeader = document.getElementById('resultHeader');
    const resultTitle = document.getElementById('resultTitle');
    const attritionProb = document.getElementById('attritionProb');
    const retentionProb = document.getElementById('retentionProb');
    const riskLevel = document.getElementById('riskLevel');
    const recommendation = document.getElementById('recommendation');
    
    // Set probabilities
    attritionProb.textContent = (result.attrition_probability * 100).toFixed(2) + '%';
    retentionProb.textContent = (result.retention_probability * 100).toFixed(2) + '%';
    
    // Set risk level and styling
    if (result.prediction === 1) {
        resultHeader.className = 'result-header danger';
        resultTitle.innerHTML = '<i class="fas fa-exclamation-circle"></i> High Attrition Risk';
        riskLevel.innerHTML = `<span class="badge bg-danger"><i class="fas fa-exclamation-triangle"></i> ${result.risk_level} Risk</span>`;
        
        recommendation.innerHTML = `
            <div class="alert alert-danger">
                <h5><i class="fas fa-lightbulb"></i> Recommendations:</h5>
                <ul>
                    <li>Immediate outreach to customer required</li>
                    <li>Consider offering retention incentives</li>
                    <li>Review customer satisfaction and service quality</li>
                    <li>Analyze transaction patterns for concerns</li>
                </ul>
            </div>
        `;
    } else {
        resultHeader.className = 'result-header success';
        resultTitle.innerHTML = '<i class="fas fa-check-circle"></i> Low Attrition Risk';
        riskLevel.innerHTML = `<span class="badge bg-success"><i class="fas fa-check"></i> ${result.risk_level} Risk</span>`;
        
        recommendation.innerHTML = `
            <div class="alert alert-success">
                <h5><i class="fas fa-lightbulb"></i> Recommendations:</h5>
                <ul>
                    <li>Continue maintaining current service quality</li>
                    <li>Consider upselling opportunities</li>
                    <li>Monitor for any changes in behavior</li>
                    <li>Maintain regular engagement</li>
                </ul>
            </div>
        `;
    }
    
    // Show results with animation
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
