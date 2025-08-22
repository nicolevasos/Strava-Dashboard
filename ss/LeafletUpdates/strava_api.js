
const auth_link = "https://www.strava.com/oauth/token"

function getActivites(res){

    const activities_link = `https://www.strava.com/api/v3/athlete/activities?access_token=${res.access_token}&per_page=200&page=1`;
    fetch(activities_link)
        .then((res) => res.json())
        .then(function (data){
            
            console.log("Total activities retrieved:", data.length); // 👈 LOG DATA LENGTH
            var map = L.map('map').setView([47.8014, 13.0448], 9);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);

            for (var x = 0; x < data.length; x++) {
                if (data[x].map.summary_polyline) {
                    console.log(data[x].map.id, data[x].type, data[x].map.summary_polyline);
                    var coordinates = L.Polyline.fromEncoded(data[x].map.summary_polyline).getLatLngs();
                    console.log(coordinates);
            
                    L.polyline(
                        coordinates,
                        {
                            color: "green",
                            weight: 5,
                            opacity: 0.7,
                            lineJoin: 'round'
                        }
                    ).addTo(map);
                }
            }
            

        }
        )
}

    
function reAuthorize() {
    fetch(auth_link, {
        method: 'post',

        headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({

            client_id: 'XXXXXXXXXXXXXXXX',
            client_secret: 'XXXXXXXXXXXXXXXX',
            refresh_token: 'XXXXXXXXXXXXXXXX',
            grant_type: 'refresh_token'

        })

    }).then(res => res.json())
        .then(res => getActivites(res))
}

reAuthorize()