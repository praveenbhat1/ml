document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const form = e.target;
    const submitBtn = form.querySelector('.action-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    const formWrapper = document.querySelector('.form-wrapper');
    const resultCard = document.getElementById('resultCard');
    const resultText = document.getElementById('resultText');
    const insightText = document.getElementById('insightText');

    // Prepare payload
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // UI Loading state
    btnText.style.display = 'none';
    loader.style.display = 'block';
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
            formWrapper.classList.add('hidden');
            resultCard.classList.remove('hidden');
            
            const rawPrediction = resData.prediction.replace(' Yield', '');
            resultText.textContent = rawPrediction;
            
            // Apply classification-specific coloring
            resultText.className = ''; // reset
            if (rawPrediction === 'High') {
                resultText.classList.add('high-yield');
                insightText.textContent = "Current classification parameters confirm a high probability of yield efficiency. Environmental alignment is optimal.";
            } else if (rawPrediction === 'Medium') {
                resultText.classList.add('medium-yield');
                insightText.textContent = "Stable yield potential identified. Standard cultivation parameters detected within expected historical variance.";
            } else {
                resultText.classList.add('low-yield');
                insightText.textContent = "A lower yield potential has been identified. Model variance suggests discrepancies in resource or climate thresholds.";
            }

            // Smooth scroll to top of main area
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
            alert("Analysis Error: " + resData.error);
        }
    } catch (error) {
        console.error("Connection error:", error);
        alert("System connection failure.");
    } finally {
        btnText.style.display = 'block';
        loader.style.display = 'none';
        submitBtn.disabled = false;
    }
});
