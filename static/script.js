// Check API health on page load
document.addEventListener('DOMContentLoaded', function() {
    checkAPIHealth();
    setupForm();
    setupSlider();
    loadCharts();
});

function checkAPIHealth() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            const statusEl = document.getElementById('status-text');
            if (data.model_loaded) {
                statusEl.textContent = '✅ API Ready';
                statusEl.parentElement.style.background = 'rgba(46, 204, 113, 0.3)';
            } else {
                statusEl.textContent = '⚠️ API Loading Model';
                statusEl.parentElement.style.background = 'rgba(243, 156, 18, 0.3)';
            }
        })
        .catch(error => {
            console.error('Health check failed:', error);
            document.getElementById('status-text').textContent = '❌ API Offline';
            document.getElementById('status-text').parentElement.style.background = 'rgba(231, 76, 60, 0.3)';
        });
}

function setupForm() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', handlePrediction);
    }
}

function setupSlider() {
    const slider = document.getElementById('avg_score');
    const valueDisplay = document.getElementById('score-value');
    
    if (slider && valueDisplay) {
        slider.addEventListener('input', function() {
            valueDisplay.textContent = this.value;
        });
    }
}

async function handlePrediction(e) {
    e.preventDefault();
    
    const submitBtn = document.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    
    try {
        // Collect form data
        const data = {
            num_clicks: parseInt(document.getElementById('num_clicks').value),
            days_active: parseInt(document.getElementById('days_active').value),
            avg_score: parseFloat(document.getElementById('avg_score').value),
            studied_credits: parseInt(document.getElementById('studied_credits').value)
        };
        
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span> Processing...';
        
        // Call API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            const result = await response.json();
            displayResults(result);
        } else {
            const error = await response.json();
            showError(error.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Connection error. Make sure FastAPI is running.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
}

function displayResults(result) {
    // Update result cards
    document.getElementById('prediction-level').textContent = result.level;
    document.getElementById('engagement-score').textContent = result.engagement_score.toFixed(4);
    document.getElementById('consistency-score').textContent = result.consistency.toFixed(4);
    
    // Show results section
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'grid';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Display message
    const messageEl = document.getElementById('result-message');
    if (result.prediction === 0) {
        messageEl.className = 'result-message error';
        messageEl.innerHTML = '⚠️ <strong>Low Performance</strong><br>This student may need additional support and resources.';
    } else if (result.prediction === 1) {
        messageEl.className = 'result-message warning';
        messageEl.innerHTML = '⚡ <strong>Medium Performance</strong><br>There is room for improvement. Consider mentoring or additional materials.';
    } else {
        messageEl.className = 'result-message success';
        messageEl.innerHTML = '✅ <strong>High Performance</strong><br>Excellent work! This student is doing great.';
    }
}

function showError(message) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'grid';
    
    const messageEl = document.getElementById('result-message');
    messageEl.className = 'result-message error';
    messageEl.textContent = '❌ Error: ' + message;
    
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function resetForm() {
    document.getElementById('prediction-form').reset();
    document.getElementById('score-value').textContent = '72.5';
    document.getElementById('results-section').style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Charts
function loadCharts() {
    const performanceCtx = document.getElementById('performanceChart');
    const engagementCtx = document.getElementById('engagementChart');
    
    if (performanceCtx) {
        new Chart(performanceCtx, {
            type: 'doughnut',
            data: {
                labels: ['Low', 'Medium', 'High'],
                datasets: [{
                    data: [25, 45, 30],
                    backgroundColor: ['#ff6b6b', '#ffd93d', '#6bcf7f'],
                    borderColor: 'white',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    if (engagementCtx) {
        new Chart(engagementCtx, {
            type: 'bar',
            data: {
                labels: ['Low', 'Medium', 'High'],
                datasets: [{
                    label: 'Avg Engagement Score',
                    data: [0.2, 0.5, 0.8],
                    backgroundColor: ['#ff6b6b', '#ffd93d', '#6bcf7f'],
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
}
