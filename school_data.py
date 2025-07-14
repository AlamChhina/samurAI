import requests

datasets = {
    'contact': 'fb3a7c18-90af-453e-bc0a-a76ecc471862',
    'enrolment': '502b8125-e48b-443a-b463-a5b76eda8c25',
    'quick_facts': 'acc1ff32-3995-469d-9e94-adedf17f9c4e'
}
base_url = 'https://open.canada.ca/data/api/3/action/package_show?id='
results = {}
for key, dataset_id in datasets.items():
    r = requests.get(base_url + dataset_id)
    data = r.json().get('result', {})
    resources = data.get('resources', [])
    results[key] = [(res.get('id'), res.get('format'), res.get('name')) for res in resources]

print(results)
