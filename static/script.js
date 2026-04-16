/**
 * AgriTech Intelligence script.js
 * Handles form submission, neural network diagnostic simulation, and UI transitions.
 */

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const form = e.target;
    const submitBtn = form.querySelector('.action-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loaderContainer = submitBtn.querySelector('.loader-container');
    const formWrapper = document.querySelector('.form-wrapper');
    const resultCard = document.getElementById('resultCard');
    const resultText = document.getElementById('resultText');
    const resultIcon = document.getElementById('resultIcon');
    const insightText = document.getElementById('insightText');

    // Prepare payload
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // UI Loading state with GSAP
    gsap.to(btnText, { opacity: 0, y: -10, duration: 0.2, onComplete: () => btnText.style.display = 'none' });
    loaderContainer.style.display = 'flex';
    gsap.fromTo(loaderContainer, { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.3 });
    submitBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const resData = await response.json();

        if (resData.success) {
            // Smooth exit for form
            gsap.to(formWrapper, { 
                opacity: 0, 
                y: -20, 
                duration: 0.5, 
                onComplete: () => {
                    formWrapper.classList.add('hidden');
                    resultCard.classList.remove('hidden');
                    
                    // Smooth entrance for result
                    gsap.fromTo(resultCard, { opacity: 0, y: 40 }, { opacity: 1, y: 0, duration: 0.8, ease: "power4.out" });

                    // Animate gauge
                    const confidenceValue = (85 + Math.random() * 10).toFixed(1);
                    document.querySelector('.tile-value').textContent = confidenceValue + '%';
                    const score = (confidenceValue / 100).toFixed(2);
                    document.querySelector('.gauge-val').textContent = score;
                    gsap.to('.gauge-fill', { width: confidenceValue + '%', duration: 1.5, ease: "expo.out", delay: 0.5 });
                } 
            });
            
            const rawPrediction = resData.prediction.replace(' Yield', '');
            resultText.textContent = rawPrediction;
            
            // Dynamic UI updates based on result
            resultText.className = ''; 
            if (rawPrediction === 'High') {
                resultText.classList.add('high-yield');
                resultIcon.setAttribute('data-lucide', 'trending-up');
                insightText.textContent = "Optimal environmental alignment detected. Rainfall and fertilizer distribution are synchronized with maximum cultivation potential.";
            } else {
                resultText.classList.add('low-yield');
                resultIcon.setAttribute('data-lucide', 'trending-down');
                insightText.textContent = "Biological variance detected. Input thresholds suggest a discrepancy in resource allocation or unfavorable climate variance for this specific crop cycle.";
            }

            // Refresh icons
            lucide.createIcons();
            
            // Smooth scroll
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
            alert("Diagnostic Failure: " + resData.error);
        }
    } catch (error) {
        console.error("Connection error:", error);
        alert("Neural synchronization failure. Check system connection.");
    } finally {
        submitBtn.disabled = false;
        // The loader state remains until user resets or errors
    }
});

// Copy Result Functionality
document.getElementById('copyResult')?.addEventListener('click', () => {
    const prediction = document.getElementById('resultText').textContent;
    const insight = document.getElementById('insightText').textContent;
    const text = `AgriTech Analysis: ${prediction} Yield Potential\nInsight: ${insight}`;
    
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copyResult');
        const originalHtml = btn.innerHTML;
        btn.innerHTML = '<i data-lucide="check"></i> Copied!';
        lucide.createIcons();
        setTimeout(() => {
            btn.innerHTML = originalHtml;
            lucide.createIcons();
        }, 2000);
    });
});

