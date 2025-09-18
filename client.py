import requests
import json

# Adres URL endpointu
url_long = "http://localhost:8081/predict/long"

# Przykładowe dane (w prawdziwym użyciu pobrałbyś je z giełdy)
# Pamiętaj, aby wysłać listę zawierającą co najmniej 50 świec dla każdego interwału.
payload = {
  "recent_1h": [
    {
      "timestamp": "2025-09-14T06:00:00",
      "open": 10.38,
      "high": 10.42,
      "low": 10.36,
      "close": 10.37,
      "volume": 128.8
    },
    {
      "timestamp": "2025-09-14T07:00:00",
      "open": 10.37,
      "high": 10.41,
      "low": 10.35,
      "close": 10.36,
      "volume": 154.5
    },
    {
      "timestamp": "2025-09-14T08:00:00",
      "open": 10.36,
      "high": 10.4,
      "low": 10.34,
      "close": 10.35,
      "volume": 133.2
    },
    {
      "timestamp": "2025-09-14T09:00:00",
      "open": 10.35,
      "high": 10.39,
      "low": 10.33,
      "close": 10.34,
      "volume": 112.9
    },
    {
      "timestamp": "2025-09-14T10:00:00",
      "open": 10.34,
      "high": 10.38,
      "low": 10.32,
      "close": 10.33,
      "volume": 145.8
    },
    {
      "timestamp": "2025-09-14T11:00:00",
      "open": 10.33,
      "high": 10.37,
      "low": 10.31,
      "close": 10.32,
      "volume": 167.1
    },
    {
      "timestamp": "2025-09-14T12:00:00",
      "open": 10.32,
      "high": 10.36,
      "low": 10.3,
      "close": 10.31,
      "volume": 189.4
    },
    {
      "timestamp": "2025-09-14T13:00:00",
      "open": 10.31,
      "high": 10.35,
      "low": 10.29,
      "close": 10.3,
      "volume": 143.7
    },
    {
      "timestamp": "2025-09-14T14:00:00",
      "open": 10.3,
      "high": 10.34,
      "low": 10.28,
      "close": 10.29,
      "volume": 156.2
    },
    {
      "timestamp": "2025-09-14T15:00:00",
      "open": 10.29,
      "high": 10.33,
      "low": 10.27,
      "close": 10.28,
      "volume": 178.5
    },
    {
      "timestamp": "2025-09-14T16:00:00",
      "open": 10.28,
      "high": 10.32,
      "low": 10.26,
      "close": 10.27,
      "volume": 199.8
    },
    {
      "timestamp": "2025-09-14T17:00:00",
      "open": 10.27,
      "high": 10.31,
      "low": 10.25,
      "close": 10.26,
      "volume": 155.1
    },
    {
      "timestamp": "2025-09-14T18:00:00",
      "open": 10.26,
      "high": 10.3,
      "low": 10.24,
      "close": 10.25,
      "volume": 167.4
    },
    {
      "timestamp": "2025-09-14T19:00:00",
      "open": 10.25,
      "high": 10.29,
      "low": 10.23,
      "close": 10.24,
      "volume": 189.7
    },
    {
      "timestamp": "2025-09-14T20:00:00",
      "open": 10.24,
      "high": 10.28,
      "low": 10.22,
      "close": 10.23,
      "volume": 211.0
    },
    {
      "timestamp": "2025-09-14T21:00:00",
      "open": 10.23,
      "high": 10.27,
      "low": 10.21,
      "close": 10.22,
      "volume": 166.3
    },
    {
      "timestamp": "2025-09-14T22:00:00",
      "open": 10.22,
      "high": 10.26,
      "low": 10.2,
      "close": 10.21,
      "volume": 178.6
    },
    {
      "timestamp": "2025-09-14T23:00:00",
      "open": 10.21,
      "high": 10.25,
      "low": 10.19,
      "close": 10.2,
      "volume": 200.9
    },
    {
      "timestamp": "2025-09-15T00:00:00",
      "open": 10.2,
      "high": 10.24,
      "low": 10.18,
      "close": 10.19,
      "volume": 223.2
    },
    {
      "timestamp": "2025-09-15T01:00:00",
      "open": 10.19,
      "high": 10.23,
      "low": 10.17,
      "close": 10.18,
      "volume": 177.5
    },
    {
      "timestamp": "2025-09-15T02:00:00",
      "open": 10.18,
      "high": 10.22,
      "low": 10.16,
      "close": 10.17,
      "volume": 189.8
    },
    {
      "timestamp": "2025-09-15T03:00:00",
      "open": 10.17,
      "high": 10.21,
      "low": 10.15,
      "close": 10.16,
      "volume": 212.1
    },
    {
      "timestamp": "2025-09-15T04:00:00",
      "open": 10.16,
      "high": 10.2,
      "low": 10.14,
      "close": 10.15,
      "volume": 234.4
    },
    {
      "timestamp": "2025-09-15T05:00:00",
      "open": 10.15,
      "high": 10.19,
      "low": 10.13,
      "close": 10.14,
      "volume": 188.7
    },
    {
      "timestamp": "2025-09-15T06:00:00",
      "open": 10.14,
      "high": 10.18,
      "low": 10.12,
      "close": 10.13,
      "volume": 201.0
    },
    {
      "timestamp": "2025-09-15T07:00:00",
      "open": 10.13,
      "high": 10.17,
      "low": 10.11,
      "close": 10.12,
      "volume": 223.3
    },
    {
      "timestamp": "2025-09-15T08:00:00",
      "open": 10.12,
      "high": 10.16,
      "low": 10.1,
      "close": 10.11,
      "volume": 245.6
    },
    {
      "timestamp": "2025-09-15T09:00:00",
      "open": 10.11,
      "high": 10.15,
      "low": 10.09,
      "close": 10.1,
      "volume": 200.0
    },
    {
      "timestamp": "2025-09-15T10:00:00",
      "open": 10.1,
      "high": 10.14,
      "low": 10.08,
      "close": 10.09,
      "volume": 212.3
    },
    {
      "timestamp": "2025-09-15T11:00:00",
      "open": 10.09,
      "high": 10.13,
      "low": 10.07,
      "close": 10.08,
      "volume": 234.6
    },
    {
      "timestamp": "2025-09-15T12:00:00",
      "open": 10.08,
      "high": 10.12,
      "low": 10.06,
      "close": 10.07,
      "volume": 256.9
    },
    {
      "timestamp": "2025-09-15T13:00:00",
      "open": 10.07,
      "high": 10.11,
      "low": 10.05,
      "close": 10.06,
      "volume": 211.2
    },
    {
      "timestamp": "2025-09-15T14:00:00",
      "open": 10.06,
      "high": 10.1,
      "low": 10.04,
      "close": 10.05,
      "volume": 223.5
    },
    {
      "timestamp": "2025-09-15T15:00:00",
      "open": 10.05,
      "high": 10.09,
      "low": 10.03,
      "close": 10.04,
      "volume": 245.8
    },
    {
      "timestamp": "2025-09-15T16:00:00",
      "open": 10.04,
      "high": 10.08,
      "low": 10.02,
      "close": 10.03,
      "volume": 268.1
    },
    {
      "timestamp": "2025-09-15T17:00:00",
      "open": 10.03,
      "high": 10.07,
      "low": 10.01,
      "close": 10.02,
      "volume": 222.4
    },
    {
      "timestamp": "2025-09-15T18:00:00",
      "open": 10.02,
      "high": 10.06,
      "low": 10.0,
      "close": 10.01,
      "volume": 234.7
    },
    {
      "timestamp": "2025-09-15T19:00:00",
      "open": 10.01,
      "high": 10.05,
      "low": 9.99,
      "close": 10.0,
      "volume": 257.0
    },
    {
      "timestamp": "2025-09-15T20:00:00",
      "open": 10.0,
      "high": 10.1,
      "low": 9.98,
      "close": 10.09,
      "volume": 279.3
    },
    {
      "timestamp": "2025-09-15T21:00:00",
      "open": 10.09,
      "high": 10.19,
      "low": 10.08,
      "close": 10.18,
      "volume": 233.6
    },
    {
      "timestamp": "2025-09-15T22:00:00",
      "open": 10.18,
      "high": 10.28,
      "low": 10.17,
      "close": 10.27,
      "volume": 245.9
    },
    {
      "timestamp": "2025-09-15T23:00:00",
      "open": 10.27,
      "high": 10.37,
      "low": 10.26,
      "close": 10.36,
      "volume": 268.2
    },
    {
      "timestamp": "2025-09-16T00:00:00",
      "open": 10.36,
      "high": 10.3,
      "low": 10.2,
      "close": 10.25,
      "volume": 290.5
    },
    {
      "timestamp": "2025-09-16T01:00:00",
      "open": 10.25,
      "high": 10.29,
      "low": 10.19,
      "close": 10.24,
      "volume": 244.8
    },
    {
      "timestamp": "2025-09-16T02:00:00",
      "open": 10.24,
      "high": 10.28,
      "low": 10.18,
      "close": 10.23,
      "volume": 257.1
    },
    {
      "timestamp": "2025-09-16T03:00:00",
      "open": 10.23,
      "high": 10.27,
      "low": 10.17,
      "close": 10.22,
      "volume": 279.4
    },
    {
      "timestamp": "2025-09-16T04:00:00",
      "open": 10.22,
      "high": 10.26,
      "low": 10.16,
      "close": 10.21,
      "volume": 301.7
    },
    {
      "timestamp": "2025-09-16T05:00:00",
      "open": 10.21,
      "high": 10.25,
      "low": 10.15,
      "close": 10.2,
      "volume": 256.0
    },
    {
      "timestamp": "2025-09-16T06:00:00",
      "open": 10.2,
      "high": 10.24,
      "low": 10.14,
      "close": 10.19,
      "volume": 268.3
    },
    {
      "timestamp": "2025-09-16T07:00:00",
      "open": 10.19,
      "high": 10.23,
      "low": 10.13,
      "close": 10.18,
      "volume": 290.6
    }
  ],
  "recent_4h": [
    {
      "timestamp": "2025-08-28T16:00:00",
      "open": 10.51,
      "high": 10.59,
      "low": 10.49,
      "close": 10.55,
      "volume": 890.3
    },
    {
      "timestamp": "2025-08-28T20:00:00",
      "open": 10.55,
      "high": 10.6,
      "low": 10.53,
      "close": 10.58,
      "volume": 765.4
    },
    {
      "timestamp": "2025-08-29T00:00:00",
      "open": 10.58,
      "high": 10.65,
      "low": 10.56,
      "close": 10.63,
      "volume": 912.7
    },
    {
      "timestamp": "2025-08-29T04:00:00",
      "open": 10.63,
      "high": 10.7,
      "low": 10.61,
      "close": 10.68,
      "volume": 843.1
    },
    {
      "timestamp": "2025-08-29T08:00:00",
      "open": 10.68,
      "high": 10.75,
      "low": 10.66,
      "close": 10.73,
      "volume": 956.4
    },
    {
      "timestamp": "2025-08-29T12:00:00",
      "open": 10.73,
      "high": 10.8,
      "low": 10.71,
      "close": 10.78,
      "volume": 1012.8
    },
    {
      "timestamp": "2025-08-29T16:00:00",
      "open": 10.78,
      "high": 10.85,
      "low": 10.76,
      "close": 10.83,
      "volume": 987.5
    },
    {
      "timestamp": "2025-08-29T20:00:00",
      "open": 10.83,
      "high": 10.9,
      "low": 10.81,
      "close": 10.88,
      "volume": 1054.2
    },
    {
      "timestamp": "2025-08-30T00:00:00",
      "open": 10.88,
      "high": 10.95,
      "low": 10.86,
      "close": 10.93,
      "volume": 1101.9
    },
    {
      "timestamp": "2025-08-30T04:00:00",
      "open": 10.93,
      "high": 11.0,
      "low": 10.91,
      "close": 10.98,
      "volume": 1032.6
    },
    {
      "timestamp": "2025-08-30T08:00:00",
      "open": 10.98,
      "high": 11.05,
      "low": 10.96,
      "close": 11.03,
      "volume": 1145.3
    },
    {
      "timestamp": "2025-08-30T12:00:00",
      "open": 11.03,
      "high": 11.1,
      "low": 11.01,
      "close": 11.08,
      "volume": 1201.7
    },
    {
      "timestamp": "2025-08-30T16:00:00",
      "open": 11.08,
      "high": 11.15,
      "low": 11.06,
      "close": 11.13,
      "volume": 1176.4
    },
    {
      "timestamp": "2025-08-30T20:00:00",
      "open": 11.13,
      "high": 11.2,
      "low": 11.11,
      "close": 11.18,
      "volume": 1243.1
    },
    {
      "timestamp": "2025-08-31T00:00:00",
      "open": 11.18,
      "high": 11.25,
      "low": 11.16,
      "close": 11.23,
      "volume": 1290.8
    },
    {
      "timestamp": "2025-08-31T04:00:00",
      "open": 11.23,
      "high": 11.3,
      "low": 11.21,
      "close": 11.28,
      "volume": 1221.5
    },
    {
      "timestamp": "2025-08-31T08:00:00",
      "open": 11.28,
      "high": 11.35,
      "low": 11.26,
      "close": 11.33,
      "volume": 1334.2
    },
    {
      "timestamp": "2025-08-31T12:00:00",
      "open": 11.33,
      "high": 11.4,
      "low": 11.31,
      "close": 11.38,
      "volume": 1390.6
    },
    {
      "timestamp": "2025-08-31T16:00:00",
      "open": 11.38,
      "high": 11.45,
      "low": 11.36,
      "close": 11.43,
      "volume": 1365.3
    },
    {
      "timestamp": "2025-08-31T20:00:00",
      "open": 11.43,
      "high": 11.5,
      "low": 11.41,
      "close": 11.48,
      "volume": 1432.0
    },
    {
      "timestamp": "2025-09-01T00:00:00",
      "open": 11.48,
      "high": 11.4,
      "low": 11.3,
      "close": 11.35,
      "volume": 1479.7
    },
    {
      "timestamp": "2025-09-02T00:00:00",
      "open": 11.35,
      "high": 11.2,
      "low": 11.1,
      "close": 11.15,
      "volume": 1400.4
    },
    {
      "timestamp": "2025-09-03T00:00:00",
      "open": 11.15,
      "high": 11.0,
      "low": 10.9,
      "close": 10.95,
      "volume": 1589.1
    },
    {
      "timestamp": "2025-09-04T00:00:00",
      "open": 10.95,
      "high": 10.8,
      "low": 10.7,
      "close": 10.75,
      "volume": 1519.8
    },
    {
      "timestamp": "2025-09-05T00:00:00",
      "open": 10.75,
      "high": 10.6,
      "low": 10.5,
      "close": 10.55,
      "volume": 1622.5
    },
    {
      "timestamp": "2025-09-06T00:00:00",
      "open": 10.55,
      "high": 10.4,
      "low": 10.3,
      "close": 10.35,
      "volume": 1669.2
    },
    {
      "timestamp": "2025-09-07T00:00:00",
      "open": 10.35,
      "high": 10.2,
      "low": 10.1,
      "close": 10.15,
      "volume": 1599.9
    },
    {
      "timestamp": "2025-09-08T00:00:00",
      "open": 10.15,
      "high": 10.0,
      "low": 9.9,
      "close": 9.95,
      "volume": 1788.6
    },
    {
      "timestamp": "2025-09-08T04:00:00",
      "open": 9.95,
      "high": 10.05,
      "low": 9.93,
      "close": 10.03,
      "volume": 1718.3
    },
    {
      "timestamp": "2025-09-08T08:00:00",
      "open": 10.03,
      "high": 10.13,
      "low": 10.01,
      "close": 10.11,
      "volume": 1821.0
    },
    {
      "timestamp": "2025-09-08T12:00:00",
      "open": 10.11,
      "high": 10.21,
      "low": 10.09,
      "close": 10.19,
      "volume": 1867.7
    },
    {
      "timestamp": "2025-09-08T16:00:00",
      "open": 10.19,
      "high": 10.29,
      "low": 10.17,
      "close": 10.27,
      "volume": 1797.4
    },
    {
      "timestamp": "2025-09-08T20:00:00",
      "open": 10.27,
      "high": 10.37,
      "low": 10.25,
      "close": 10.35,
      "volume": 1900.1
    },
    {
      "timestamp": "2025-09-09T00:00:00",
      "open": 10.35,
      "high": 10.45,
      "low": 10.33,
      "close": 10.43,
      "volume": 1946.8
    },
    {
      "timestamp": "2025-09-10T00:00:00",
      "open": 10.43,
      "high": 10.53,
      "low": 10.41,
      "close": 10.51,
      "volume": 1876.5
    },
    {
      "timestamp": "2025-09-11T00:00:00",
      "open": 10.51,
      "high": 10.61,
      "low": 10.49,
      "close": 10.59,
      "volume": 1979.2
    },
    {
      "timestamp": "2025-09-12T00:00:00",
      "open": 10.59,
      "high": 10.69,
      "low": 10.57,
      "close": 10.67,
      "volume": 2025.9
    },
    {
      "timestamp": "2025-09-13T00:00:00",
      "open": 10.67,
      "high": 10.77,
      "low": 10.65,
      "close": 10.75,
      "volume": 1955.6
    },
    {
      "timestamp": "2025-09-14T00:00:00",
      "open": 10.75,
      "high": 10.85,
      "low": 10.73,
      "close": 10.83,
      "volume": 2058.3
    },
    {
      "timestamp": "2025-09-14T04:00:00",
      "open": 10.83,
      "high": 10.93,
      "low": 10.81,
      "close": 10.91,
      "volume": 2105.0
    },
    {
      "timestamp": "2025-09-14T08:00:00",
      "open": 10.91,
      "high": 11.01,
      "low": 10.89,
      "close": 10.99,
      "volume": 2034.7
    },
    {
      "timestamp": "2025-09-14T12:00:00",
      "open": 10.99,
      "high": 11.09,
      "low": 10.97,
      "close": 11.07,
      "volume": 2137.4
    },
    {
      "timestamp": "2025-09-14T16:00:00",
      "open": 11.07,
      "high": 11.17,
      "low": 11.05,
      "close": 11.15,
      "volume": 2184.1
    },
    {
      "timestamp": "2025-09-14T20:00:00",
      "open": 11.15,
      "high": 11.25,
      "low": 11.13,
      "close": 11.23,
      "volume": 2113.8
    },
    {
      "timestamp": "2025-09-15T00:00:00",
      "open": 11.23,
      "high": 11.33,
      "low": 11.21,
      "close": 11.31,
      "volume": 2216.5
    },
    {
      "timestamp": "2025-09-15T04:00:00",
      "open": 11.31,
      "high": 11.41,
      "low": 11.29,
      "close": 11.39,
      "volume": 2263.2
    },
    {
      "timestamp": "2025-09-15T08:00:00",
      "open": 11.39,
      "high": 11.49,
      "low": 11.37,
      "close": 11.47,
      "volume": 2192.9
    },
    {
      "timestamp": "2025-09-15T12:00:00",
      "open": 11.47,
      "high": 11.57,
      "low": 11.45,
      "close": 11.55,
      "volume": 2295.6
    },
    {
      "timestamp": "2025-09-15T16:00:00",
      "open": 11.55,
      "high": 11.6,
      "low": 11.5,
      "close": 11.55,
      "volume": 2342.3
    },
    {
      "timestamp": "2025-09-15T20:00:00",
      "open": 11.55,
      "high": 11.4,
      "low": 11.3,
      "close": 11.35,
      "volume": 2272.0
    },
    {
      "timestamp": "2025-09-16T00:00:00",
      "open": 11.35,
      "high": 11.2,
      "low": 11.1,
      "close": 11.15,
      "volume": 2374.7
    },
    {
      "timestamp": "2025-09-16T04:00:00",
      "open": 11.15,
      "high": 11.0,
      "low": 10.9,
      "close": 10.95,
      "volume": 2421.4
    }
  ],
  "recent_1d": [
    {
      "timestamp": "2025-07-29T00:00:00",
      "open": 10.89,
      "high": 11.05,
      "low": 10.85,
      "close": 10.99,
      "volume": 25432.1
    },
    {
      "timestamp": "2025-07-30T00:00:00",
      "open": 10.99,
      "high": 11.1,
      "low": 10.95,
      "close": 11.05,
      "volume": 23145.9
    },
    {
      "timestamp": "2025-07-31T00:00:00",
      "open": 11.05,
      "high": 11.2,
      "low": 11.0,
      "close": 11.15,
      "volume": 26789.5
    },
    {
      "timestamp": "2025-08-01T00:00:00",
      "open": 11.15,
      "high": 11.3,
      "low": 11.1,
      "close": 11.25,
      "volume": 24321.8
    },
    {
      "timestamp": "2025-08-02T00:00:00",
      "open": 11.25,
      "high": 11.4,
      "low": 11.2,
      "close": 11.35,
      "volume": 27984.3
    },
    {
      "timestamp": "2025-08-03T00:00:00",
      "open": 11.35,
      "high": 11.5,
      "low": 11.3,
      "close": 11.45,
      "volume": 25543.2
    },
    {
      "timestamp": "2025-08-04T00:00:00",
      "open": 11.45,
      "high": 11.6,
      "low": 11.4,
      "close": 11.55,
      "volume": 29123.6
    },
    {
      "timestamp": "2025-08-05T00:00:00",
      "open": 11.55,
      "high": 11.7,
      "low": 11.5,
      "close": 11.65,
      "volume": 26789.1
    },
    {
      "timestamp": "2025-08-06T00:00:00",
      "open": 11.65,
      "high": 11.8,
      "low": 11.6,
      "close": 11.75,
      "volume": 30543.2
    },
    {
      "timestamp": "2025-08-07T00:00:00",
      "open": 11.75,
      "high": 11.9,
      "low": 11.7,
      "close": 11.85,
      "volume": 28321.5
    },
    {
      "timestamp": "2025-08-08T00:00:00",
      "open": 11.85,
      "high": 12.0,
      "low": 11.8,
      "close": 11.95,
      "volume": 31987.4
    },
    {
      "timestamp": "2025-08-09T00:00:00",
      "open": 11.95,
      "high": 12.1,
      "low": 11.9,
      "close": 12.05,
      "volume": 29876.3
    },
    {
      "timestamp": "2025-08-10T00:00:00",
      "open": 12.05,
      "high": 12.2,
      "low": 12.0,
      "close": 12.15,
      "volume": 33456.7
    },
    {
      "timestamp": "2025-08-11T00:00:00",
      "open": 12.15,
      "high": 12.3,
      "low": 12.1,
      "close": 12.25,
      "volume": 31234.5
    },
    {
      "timestamp": "2025-08-12T00:00:00",
      "open": 12.25,
      "high": 12.4,
      "low": 12.2,
      "close": 12.35,
      "volume": 34876.9
    },
    {
      "timestamp": "2025-08-13T00:00:00",
      "open": 12.35,
      "high": 12.5,
      "low": 12.3,
      "close": 12.45,
      "volume": 32654.3
    },
    {
      "timestamp": "2025-08-14T00:00:00",
      "open": 12.45,
      "high": 12.6,
      "low": 12.4,
      "close": 12.55,
      "volume": 36234.1
    },
    {
      "timestamp": "2025-08-15T00:00:00",
      "open": 12.55,
      "high": 12.7,
      "low": 12.5,
      "close": 12.65,
      "volume": 34012.8
    },
    {
      "timestamp": "2025-08-16T00:00:00",
      "open": 12.65,
      "high": 12.8,
      "low": 12.6,
      "close": 12.75,
      "volume": 37890.2
    },
    {
      "timestamp": "2025-08-17T00:00:00",
      "open": 12.75,
      "high": 12.9,
      "low": 12.7,
      "close": 12.85,
      "volume": 35678.6
    },
    {
      "timestamp": "2025-08-18T00:00:00",
      "open": 12.85,
      "high": 13.0,
      "low": 12.8,
      "close": 12.95,
      "volume": 39543.1
    },
    {
      "timestamp": "2025-08-19T00:00:00",
      "open": 12.95,
      "high": 13.1,
      "low": 12.9,
      "close": 13.05,
      "volume": 37432.9
    },
    {
      "timestamp": "2025-08-20T00:00:00",
      "open": 13.05,
      "high": 13.2,
      "low": 13.0,
      "close": 13.15,
      "volume": 41234.5
    },
    {
      "timestamp": "2025-08-21T00:00:00",
      "open": 13.15,
      "high": 13.3,
      "low": 13.1,
      "close": 13.25,
      "volume": 39012.3
    },
    {
      "timestamp": "2025-08-22T00:00:00",
      "open": 13.25,
      "high": 13.4,
      "low": 13.2,
      "close": 13.35,
      "volume": 42890.7
    },
    {
      "timestamp": "2025-08-23T00:00:00",
      "open": 13.35,
      "high": 13.5,
      "low": 13.3,
      "close": 13.45,
      "volume": 40678.1
    },
    {
      "timestamp": "2025-08-24T00:00:00",
      "open": 13.45,
      "high": 13.6,
      "low": 13.4,
      "close": 13.55,
      "volume": 44543.6
    },
    {
      "timestamp": "2025-08-25T00:00:00",
      "open": 13.55,
      "high": 13.7,
      "low": 13.5,
      "close": 13.65,
      "volume": 42432.4
    },
    {
      "timestamp": "2025-08-26T00:00:00",
      "open": 13.65,
      "high": 13.8,
      "low": 13.6,
      "close": 13.75,
      "volume": 46234.8
    },
    {
      "timestamp": "2025-08-27T00:00:00",
      "open": 13.75,
      "high": 13.9,
      "low": 13.7,
      "close": 13.85,
      "volume": 44012.2
    },
    {
      "timestamp": "2025-08-28T00:00:00",
      "open": 13.85,
      "high": 14.0,
      "low": 13.8,
      "close": 13.95,
      "volume": 47890.6
    },
    {
      "timestamp": "2025-08-29T00:00:00",
      "open": 13.95,
      "high": 14.1,
      "low": 13.9,
      "close": 14.05,
      "volume": 45678.0
    },
    {
      "timestamp": "2025-08-30T00:00:00",
      "open": 14.05,
      "high": 14.2,
      "low": 14.0,
      "close": 14.15,
      "volume": 49543.5
    },
    {
      "timestamp": "2025-08-31T00:00:00",
      "open": 14.15,
      "high": 14.3,
      "low": 14.1,
      "close": 14.25,
      "volume": 47432.3
    },
    {
      "timestamp": "2025-09-01T00:00:00",
      "open": 14.25,
      "high": 14.1,
      "low": 14.0,
      "close": 14.05,
      "volume": 51234.7
    },
    {
      "timestamp": "2025-09-02T00:00:00",
      "open": 14.05,
      "high": 13.9,
      "low": 13.8,
      "close": 13.85,
      "volume": 49012.1
    },
    {
      "timestamp": "2025-09-03T00:00:00",
      "open": 13.85,
      "high": 13.7,
      "low": 13.6,
      "close": 13.65,
      "volume": 52890.5
    },
    {
      "timestamp": "2025-09-04T00:00:00",
      "open": 13.65,
      "high": 13.5,
      "low": 13.4,
      "close": 13.45,
      "volume": 50678.9
    },
    {
      "timestamp": "2025-09-05T00:00:00",
      "open": 13.45,
      "high": 13.3,
      "low": 13.2,
      "close": 13.25,
      "volume": 54543.4
    },
    {
      "timestamp": "2025-09-06T00:00:00",
      "open": 13.25,
      "high": 13.1,
      "low": 13.0,
      "close": 13.05,
      "volume": 52432.2
    },
    {
      "timestamp": "2025-09-07T00:00:00",
      "open": 13.05,
      "high": 12.9,
      "low": 12.8,
      "close": 12.85,
      "volume": 56234.6
    },
    {
      "timestamp": "2025-09-08T00:00:00",
      "open": 12.85,
      "high": 12.7,
      "low": 12.6,
      "close": 12.65,
      "volume": 54012.0
    },
    {
      "timestamp": "2025-09-09T00:00:00",
      "open": 12.65,
      "high": 12.5,
      "low": 12.4,
      "close": 12.45,
      "volume": 57890.4
    },
    {
      "timestamp": "2025-09-10T00:00:00",
      "open": 12.45,
      "high": 12.3,
      "low": 12.2,
      "close": 12.25,
      "volume": 55678.8
    },
    {
      "timestamp": "2025-09-11T00:00:00",
      "open": 12.25,
      "high": 12.1,
      "low": 12.0,
      "close": 12.05,
      "volume": 59543.3
    },
    {
      "timestamp": "2025-09-12T00:00:00",
      "open": 12.05,
      "high": 11.9,
      "low": 11.8,
      "close": 11.85,
      "volume": 57432.1
    },
    {
      "timestamp": "2025-09-13T00:00:00",
      "open": 11.85,
      "high": 11.7,
      "low": 11.6,
      "close": 11.65,
      "volume": 61234.5
    },
    {
      "timestamp": "2025-09-14T00:00:00",
      "open": 11.65,
      "high": 11.5,
      "low": 11.4,
      "close": 11.45,
      "volume": 59012.9
    },
    {
      "timestamp": "2025-09-15T00:00:00",
      "open": 11.45,
      "high": 11.3,
      "low": 11.2,
      "close": 11.25,
      "volume": 62890.3
    }
  ]
}

try:
    # Wysłanie zapytania POST z danymi JSON
    response = requests.post(url_long, json=payload)
    response.raise_for_status()  # Rzuci wyjątkiem dla błędów HTTP (np. 4xx, 5xx)

    # Wyświetlenie odpowiedzi
    prediction_result = response.json()
    print("Odpowiedź serwera:")
    print(json.dumps(prediction_result, indent=2))

except requests.exceptions.RequestException as e:
    print(f"Wystąpił błąd podczas komunikacji z API: {e}")