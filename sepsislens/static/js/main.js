/* SepsisLens Frontend Scripts */
document.addEventListener('DOMContentLoaded', function () {
    // Enable Bootstrap tooltips if any
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // File input drag & drop feedback
    const fileInput = document.getElementById('csv_file_input');
    const dropzone = document.querySelector('.upload-dropzone');
    if (fileInput && dropzone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('border-teal', 'bg-light');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('border-teal', 'bg-light');
            }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                const filenameElem = document.getElementById('selected_filename');
                if (filenameElem) {
                    filenameElem.textContent = "Selected: " + files[0].name;
                    filenameElem.classList.remove('d-none');
                }
            }
        });

        fileInput.addEventListener('change', function () {
            const filenameElem = document.getElementById('selected_filename');
            if (filenameElem && fileInput.files.length > 0) {
                filenameElem.textContent = "Selected: " + fileInput.files[0].name;
                filenameElem.classList.remove('d-none');
            }
        });
    }

    // Quick filter table records
    const searchInput = document.getElementById('table_search_input');
    if (searchInput) {
        searchInput.addEventListener('keyup', function () {
            const query = this.value.toLowerCase();
            const rows = document.querySelectorAll('#results_table tbody tr');
            rows.forEach(row => {
                const text = row.innerText.toLowerCase();
                row.style.display = text.includes(query) ? '' : 'none';
            });
        });
    }
});
