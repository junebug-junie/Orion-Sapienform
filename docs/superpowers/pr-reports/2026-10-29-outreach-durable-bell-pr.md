# Orion's messages stay in the tray until Juniper dismisses them

## Bell sketch (written before the frontend change)

```
NOTIFICATIONS  [Orion 2]           Notify settings  [Dismiss all]  [All v]
+-- notificationList -----------------------------------------------------+
| Orion  10:42                                         [Reply] [Dismiss]  |
|   I noticed the cabinet ran hot again while you were out...             |
| Orion  08:15                                         [Reply] [Dismiss]  |
|   Something from the reading earlier stuck with me...                   |
| ----------------------------------------------------------------------  |
| (normal notification cards, as today)                                   |
| 966 self-cleared alerts hidden                                          |
+-------------------------------------------------------------------------+
```

- No new panel. The Orion rows sit at the top of the existing list.
- `[Orion 2]` is a small badge in the existing header. It hides at zero.
- Reply: put the chat on that conversation, show the message there, focus
  the input. Juniper types.
- Dismiss: saved in orion-notify. The row does not come back after a reload or
  a Hub restart.

(The rest of this report is filled in below.)
