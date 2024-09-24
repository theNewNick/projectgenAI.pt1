document.getElementById('uploadForm').addEventListener('submit', function(e) {
  e.preventDefault();  // Prevent the default form submission behavior
  alert('Retrieving the comprehensive report from the GenAI system...');

  const form = document.getElementById('uploadForm');
  const formData = new FormData(form);  // Gather all the form data (files included)

  // Example AJAX call to send files and retrieve the report from your Flask backend
  fetch('/analyze', {
    method: 'POST',
    body: formData
  })
  .then(response => response.blob())  // Treat the response as a blob (e.g., for downloading a file)
  .then(blob => {
      // Create a temporary URL for the blob
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = 'report.pdf';  // Specify the file name to be downloaded (e.g., 'report.pdf')
      
      // Append the anchor to the body, trigger the click to download the file, and remove the anchor
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);  // Revoke the object URL after the download
      document.body.removeChild(a);  // Clean up the DOM by removing the temporary anchor
  })
  .catch(error => {
    console.error('Error retrieving report:', error);
    alert('An error occurred while retrieving the report.');
  });
});
