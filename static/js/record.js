function startRecording() {
    fetch('/start_recording', { method: 'POST' })
    .then(response => response.json())
    .then(data => console.log(data.status));
}

function stopRecording() {
    fetch('/stop_recording', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        document.getElementById('response_area').innerText = data.response;
    });
}
