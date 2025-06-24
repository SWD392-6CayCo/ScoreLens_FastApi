# API Documentation

## Kafka Message API

### 🔸 Endpoint
`POST /kafka-messages/`

### 🔸 Example JSON for log_request
```json
{
  "code": "LOGGING",
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
    "message": "Player 6 potted ball 1",
    "details": {
      "playerID": 6,
      "gameSetID": 82,
      "scoreValue": true,
      "isFoul": false,
      "isUncertain": false,
      "message": "No foul"
    }
  }
}
