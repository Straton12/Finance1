// Initialize all charts
document.addEventListener('DOMContentLoaded', function () {
    // Banking Services Usage Chart
    fetch('/api/banking-services/')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('bankingServicesChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: 'Banking Services Usage',
                        data: data.data,
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function (value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
        });

    // Digital Services Chart
    fetch('/api/digital-services/')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('digitalServicesChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: 'Digital Services Usage',
                        data: data.data,
                        backgroundColor: 'rgba(129, 140, 248, 0.7)',
                        borderColor: 'rgba(129, 140, 248, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function (value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
        });

    // Loan Sources Chart
    fetch('/api/loan-sources/')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('loanSourcesChart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.labels,
                    datasets: [{
                        data: data.data,
                        backgroundColor: [
                            'rgba(99, 102, 241, 0.7)',
                            'rgba(129, 140, 248, 0.7)',
                            'rgba(165, 180, 252, 0.7)',
                            'rgba(199, 210, 254, 0.7)',
                            'rgba(224, 231, 255, 0.7)',
                            'rgba(238, 242, 255, 0.7)'
                        ],
                        borderColor: [
                            'rgba(99, 102, 241, 1)',
                            'rgba(129, 140, 248, 1)',
                            'rgba(165, 180, 252, 1)',
                            'rgba(199, 210, 254, 1)',
                            'rgba(224, 231, 255, 1)',
                            'rgba(238, 242, 255, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        });

    // Credit Types Chart
    fetch('/api/credit-types/')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('creditTypesChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: 'Credit Types Usage',
                        data: data.data,
                        backgroundColor: 'rgba(165, 180, 252, 0.7)',
                        borderColor: 'rgba(165, 180, 252, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function (value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
        });
}); 