import requests
import json
from math import radians, sin, cos, sqrt, atan2
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# Hàm tính khoảng cách Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Bán kính Trái Đất (km)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# API đề xuất món ăn
@app.route('/recommendMeals', methods=['GET'])
def recommend_food():
    user_id = request.args.get('user_id')

    # Lấy vị trí và default cuisine của user
    user_data_url = f"http://localhost:5110/api/Patron/GetUserAddressesAndDefaultCuisine/{user_id}"
    user_data = requests.get(user_data_url).json()
    user_geo = user_data[0]['geoLocation']
    user_lat, user_lon = map(float, user_geo.split(','))
    user_cuisines = user_data[0]['defaultCuisine'].split(', ')

    # Lấy danh sách cửa hàng
    restaurant_url = "http://localhost:5110/api/Meals/GeoLocation"
    restaurants = requests.get(restaurant_url).json()

    recommendations = []

    vectorizer = TfidfVectorizer()
    cuisine_vector = vectorizer.fit_transform(user_cuisines)

    for restaurant in restaurants:
        address = restaurant.get('address')
        if not address or 'geoLocation' not in address:
            continue  # Bỏ qua nếu không có geoLocation

        try:
            rest_lat, rest_lon = map(float, address['geoLocation'].split(','))
            distance = haversine(user_lat, user_lon, rest_lat, rest_lon)
        except ValueError:
            continue  # Bỏ qua nếu geoLocation không hợp lệ

        for menu in restaurant.get('menus', []):
            for item in menu.get('menu_items', []):
                item_vector = vectorizer.transform([item['item_name']])
                similarity = cosine_similarity(item_vector, cuisine_vector).max()

                if similarity > 0.1:  # Ngưỡng tối thiểu để coi là phù hợp
                    recommendations.append({
                        'restaurant_id': restaurant['user_id'],
                        'restaurant_name': restaurant['user_name'],
                        'menu_id': menu['menu_id'],
                        'menu_name': menu['menu_name'],
                        'item_id': item['item_id'],
                        'item_name': item['item_name'],
                        'price': item['price'],
                        'distance_km': round(distance, 2),
                        'similarity_score': round(similarity, 2)
                    })

    # Sắp xếp theo điểm số tương đồng (giảm dần) và khoảng cách gần nhất (tăng dần)
    recommendations.sort(key=lambda x: (-x['similarity_score'], x['distance_km']))

    return jsonify(recommendations[:6])  # Lấy top 6 món ăn


# API đề xuất nhà hàng
@app.route('/recommendRestaurants', methods=['GET'])
def recommend_restaurants():
    user_id = request.args.get('user_id')

    # Lấy vị trí và default cuisine của user
    user_data_url = f"http://localhost:5110/api/Patron/GetUserAddressesAndDefaultCuisine/{user_id}"
    user_data = requests.get(user_data_url).json()
    user_geo = user_data[0]['geoLocation']
    user_lat, user_lon = map(float, user_geo.split(','))
    user_cuisines = user_data[0]['defaultCuisine'].split(', ')

    # Lấy danh sách cửa hàng
    restaurant_url = "http://localhost:5110/api/Meals/GeoLocation"
    restaurants = requests.get(restaurant_url).json()

    recommendations = []

    vectorizer = TfidfVectorizer()
    cuisine_vector = vectorizer.fit_transform(user_cuisines)

    for restaurant in restaurants:
        address = restaurant.get('address')
        if not address or 'geoLocation' not in address:
            continue  # Bỏ qua nếu không có geoLocation

        try:
            rest_lat, rest_lon = map(float, address['geoLocation'].split(','))
            distance = haversine(user_lat, user_lon, rest_lat, rest_lon)
        except ValueError:
            continue  # Bỏ qua nếu geoLocation không hợp lệ

        menu_items = [item['item_name'] for menu in restaurant.get('menus', []) for item in menu.get('menu_items', [])]
        if menu_items:
            item_vector = vectorizer.transform(menu_items)
            similarity = cosine_similarity(item_vector, cuisine_vector).max()
        else:
            similarity = 0

        recommendations.append({
            'restaurant_id': restaurant['user_id'],
            'restaurant_name': restaurant['user_name'],
            'distance_km': round(distance, 2),
            'similarity_score': round(similarity, 2)
        })

    # Sắp xếp theo điểm số tương đồng (giảm dần) và khoảng cách gần nhất (tăng dần)
    recommendations.sort(key=lambda x: (-x['similarity_score'], x['distance_km']))

    return jsonify(recommendations[:6])  # Lấy top 6 nhà hàng


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5120)


