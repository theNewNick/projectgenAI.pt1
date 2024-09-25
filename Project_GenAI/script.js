document.getElementById('uploadForm').addEventListener('submit', function(e) {
  e.preventDefault();  // Prevent the default form submission behavior
  alert('Retrieving the comprehensive report from the GenAI system...');

  const form = document.getElementById('uploadForm');
  const formData = new FormData(form);  // Gather all the form data (files + inputs)

  // Fetch request to send the form data to the Flask backend
  fetch('/analyze', {
      method: 'POST',
      body: formData
  })
  .then(response => {
      if (!response.ok) {
          // Attempt to read the error message from the response
          return response.text().then(text => {
              throw new Error(text || 'Network response was not ok');
          });
      }
      // Check if the response is a PDF
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/pdf')) {
          return response.text().then(text => {
              throw new Error('Expected a PDF file, but received:\n' + text);
          });
      }
      return response.blob();  // Expecting the report as a file (e.g., PDF)
  })
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
      alert('An error occurred while retrieving the report:\n' + error.message);
  });
});