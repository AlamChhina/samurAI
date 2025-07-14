import requests

params = {
    'resource_id': '07586958-ffd4-4293-aec0-3b32e0f7de7b',
    'filters': {'city': 'Toronto', 'school_level': 'Secondary'},
    'limit': 5
}
response = requests.get('https://open.canada.ca/data/en/api/3/action/datastore_search', params=params).json()

print(response)