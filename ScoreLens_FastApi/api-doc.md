# API Documentation

## Kafka Message API

### 🔸 Endpoint
`POST /kafka-messages/`

### 🔸 Example JSON for log_request
```json
{
  "code": "LOGGING",
  "tableID": "23374e21-2391-41b0-b275-651df88b3b04",
  "data": {
    "level": "easy",
    "type": "score_create",
    "cueBallId": 0,
    "balls": [
      {
        "start": [1, 2],
        "end": [2, 3],
        "potted": false
      },
      {
        "start": [1, 8],
        "end": [2, 5],
        "potted": true
      }
    ],
    "collisions": [
      {
        "ball1": 0,
        "ball2": 1,
        "time": 1.25
      }
    ],
    "message": "Player {} potted ball {}",
    "details": {
      "playerID": 290,
      "gameSetID": 242,
      "scoreValue": true,
      "isFoul": false,
      "isUncertain": false,
      "message": "No foul"
    }
  }
}
