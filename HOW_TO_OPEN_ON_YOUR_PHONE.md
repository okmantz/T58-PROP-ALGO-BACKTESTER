# Using the T58 Prop Algo Backtester on your phone

This gives you the backtester on your Samsung (or any phone) as a
home-screen app, with no Google Play, no app store review, and no
third-party hosting account. The only requirement: your PC and your
phone need to be on the **same Wi-Fi network** whenever you use it.

## One-time setup

1. On your **Windows PC**, download `T58-Web-App-Windows.zip` from the
   GitHub Releases page and extract it anywhere (Desktop is fine).
2. Double-click **`T58-Web-App.exe`** inside the extracted folder.
   - Windows may show a blue "Windows protected your PC" SmartScreen
     warning because the exe isn't code-signed. Click **"More info"**
     then **"Run anyway"**. This is expected and safe — it's the same
     warning your desktop backtester exe shows.
3. A console window opens and, a moment later, **a QR code picture pops
   up** on your PC screen. The console also prints a web address like
   `http://192.168.1.42:5000`.
4. On your **Samsung phone**, open the Camera app and point it at the
   QR code on your PC screen. Tap the notification/link that appears —
   this opens the backtester in Chrome.
   - If the QR scan doesn't work for any reason, just type the address
     printed in the console (e.g. `http://192.168.1.42:5000`) into
     your phone's browser instead.
5. In Chrome, tap the **⋮ menu** (top right) → **"Add to Home Screen"**
   (or "Install app" if Chrome offers it directly). Confirm.

You now have a **T58 Backtester icon on your home screen**, just like a
real app. Tapping it opens the tool full-screen, no browser bar.

## Every time after that

1. On your PC, double-click `T58-Web-App.exe` again (keep the console
   window open while you use the app on your phone — closing it stops
   the backtester).
2. On your phone, tap the home-screen icon you already installed. It
   will load the tool from your PC over Wi-Fi.

That's it — no accounts, no re-scanning the QR code, no reinstalling.

## Good to know

- Your phone is just a *remote screen* for the backtester running on
  your PC — the actual number-crunching happens on your PC, same as
  the desktop app. Your PC needs to stay on and awake while you use it
  from your phone.
- This only works over your home/office Wi-Fi (or a personal hotspot
  connecting both devices) — it will not work over separate cellular
  connections, since your phone needs to actually reach your PC.
- Reports you generate are saved next to `T58-Web-App.exe` in a
  `reports` folder on your PC, exactly like the desktop app.
