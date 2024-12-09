document.getElementById('uploadForm').addEventListener('submit', function(e) {
  e.preventDefault();  // Prevent the default form submission behavior
  alert('Retrieving the comprehensive report from the GenAI system...');

  const form = document.getElementById('uploadForm');
  const formData = new FormData(form);  // Gather all the form data (files + inputs)

  // Example: Log the values to check if they are being captured correctly (optional)
  console.log('WACC:', formData.get('wacc'));
  console.log('Tax Rate:', formData.get('tax_rate'));
  console.log('Growth Rate:', formData.get('growth_rate'));
  console.log('Current Stock Price:', formData.get('stock_price'));
  console.log('Debt to Equity Benchmark:', formData.get('debt_equity_benchmark'));
  console.log('P/E Ratio Benchmark:', formData.get('pe_benchmark'));

  // Fetch request to send the form data to the Flask backend
  fetch('/analyze', {
      method: 'POST',
      body: formData
  })
  .then(response => response.blob())  // Expecting the report as a file (e.g., PDF)
  .then(blob => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = 'financial_report.pdf';  // Set the download name for the report
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
  })
  .catch(error => {
      console.error('Error retrieving report:', error);
      alert('An error occurred while retrieving the report.');
  });
});
