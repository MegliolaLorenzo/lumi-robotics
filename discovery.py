"""
Unitree Go2 — Network Discovery
Scans the local WiFi network and finds the robot's IP address.

Usage:
    python discovery.py              # scan and print results
    python discovery.py --wait       # keep scanning until the robot is found
"""

import socket
import subprocess
import sys
import time
import platform
import ipaddress
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Unitree Go2 known ports
# 8080 = main robot interface
# 8081 = WebRTC SDP signaling
# 9991 = WebRTC notify
# 9000 = alternative web interface
UNITREE_PORTS = [8080, 8081, 9991, 9000]

# Known hostname patterns for Go2
UNITREE_HOSTNAMES = ["unitree", "go2", "robot"]

# MAC prefixes used by Unitree Go2 (Realtek/Espressif WiFi chips)
UNITREE_MAC_PREFIXES = [
    "b0:a7:b9",  # Espressif (Go2 WiFi chip)
    "7c:df:a1",  # Realtek
    "00:0a:f5",  # Unitree registered
]


def get_local_network() -> str:
    """Returns the subnet of the current local network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        parts = local_ip.rsplit(".", 1)
        return f"{parts[0]}.0/24"
    except Exception:
        return "192.168.1.0/24"


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def ping_host(ip: str, timeout: float = 0.5) -> bool:
    try:
        # -W timeout flag differs by OS:
        #   macOS: milliseconds   Linux: seconds
        if platform.system() == "Darwin":
            cmd = ["ping", "-c", "1", "-W", "500", str(ip)]
        else:
            cmd = ["ping", "-c", "1", "-W", "1", str(ip)]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 1.0
        )
        return result.returncode == 0
    except Exception:
        return False


def check_port(ip: str, port: int, timeout: float = 0.8) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((str(ip), port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_unitree_ports(ip: str) -> list[int]:
    open_ports = []
    for port in UNITREE_PORTS:
        if check_port(ip, port):
            open_ports.append(port)
    return open_ports


def get_arp_table() -> dict[str, str]:
    arp_map = {}
    try:
        result = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                ip = parts[1].strip("()")
                mac = parts[3].lower() if len(parts) > 3 else ""
                if mac and mac != "(incomplete)":
                    arp_map[ip] = mac
    except Exception:
        pass
    return arp_map


def is_unitree_mac(mac: str) -> bool:
    for prefix in UNITREE_MAC_PREFIXES:
        if mac.startswith(prefix):
            return True
    return False


def try_resolve_hostname(ip: str) -> str:
    try:
        hostname = socket.gethostbyaddr(str(ip))[0].lower()
        return hostname
    except Exception:
        return ""


def is_unitree_hostname(hostname: str) -> bool:
    return any(pattern in hostname for pattern in UNITREE_HOSTNAMES)


def scan_host(ip: str) -> dict | None:
    """
    Scans a single IP.
    Returns info if it looks like a Unitree Go2, None otherwise.
    """
    ip_str = str(ip)
    evidence = []
    confidence = 0

    open_ports = check_unitree_ports(ip_str)
    if open_ports:
        confidence += 40 * len(open_ports)
        evidence.append(f"open ports: {open_ports}")

    if not open_ports and not ping_host(ip_str):
        return None

    arp = get_arp_table()
    mac = arp.get(ip_str, "")
    if mac and is_unitree_mac(mac):
        confidence += 50
        evidence.append(f"Unitree MAC: {mac}")

    hostname = try_resolve_hostname(ip_str)
    if hostname and is_unitree_hostname(hostname):
        confidence += 30
        evidence.append(f"hostname: {hostname}")

    if confidence > 0 or open_ports:
        return {
            "ip": ip_str,
            "confidence": confidence,
            "evidence": evidence,
            "open_ports": open_ports,
            "mac": mac,
            "hostname": hostname
        }
    return None


def discover_robot(subnet: str = None, max_workers: int = 50) -> list[dict]:
    """
    Scans the subnet and looks for the Unitree Go2.
    Returns a list of candidates sorted by confidence.
    """
    if subnet is None:
        subnet = get_local_network()

    network = ipaddress.IPv4Network(subnet, strict=False)
    hosts = list(network.hosts())

    print(f"  Scanning network {subnet} ({len(hosts)} hosts)...")
    print(f"  Your IP: {get_local_ip()}")
    print(f"  Looking for ports: {UNITREE_PORTS}")
    print()

    candidates = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_host, ip): ip for ip in hosts}

        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                candidates.append(result)
                print(f"  Candidate found: {result['ip']} "
                      f"(confidence: {result['confidence']}) — {result['evidence']}")

            if done % 50 == 0:
                pct = done / len(hosts) * 100
                print(f"  Progress: {done}/{len(hosts)} ({pct:.0f}%)")

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates


def find_robot_stream_url(robot_ip: str) -> str | None:
    candidates = [
        f"http://{robot_ip}:8080/api/video/stream",
        f"http://{robot_ip}:8082/video",
        f"http://{robot_ip}:9000/stream",
    ]

    for url in candidates:
        try:
            import requests
            r = requests.get(url, timeout=3, stream=True)
            if r.status_code == 200:
                print(f"  Stream found: {url}")
                return url
        except Exception:
            pass

    return None


def wait_for_robot(subnet: str = None, interval: float = 5.0) -> dict:
    """Keeps scanning until the robot is found on the network."""
    print("  Waiting for robot on the network...")
    while True:
        candidates = discover_robot(subnet)
        if candidates:
            best = candidates[0]
            print(f"\n  Robot found! IP: {best['ip']}")
            return best
        print(f"  Robot not found. Retrying in {interval:.0f}s...")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree Go2 Network Discovery")
    parser.add_argument("--wait", action="store_true",
                        help="Keep scanning until the robot is found")
    parser.add_argument("--subnet", type=str, default=None,
                        help="Subnet to scan (e.g. 192.168.1.0/24)")
    args = parser.parse_args()

    print("=" * 50)
    print("  Fight Detection — Robot Discovery")
    print("=" * 50)
    print()

    if args.wait:
        result = wait_for_robot(args.subnet)
    else:
        candidates = discover_robot(args.subnet)

        print()
        print("=" * 50)
        if not candidates:
            print("  No Unitree robot found on the network.")
            print("  Make sure:")
            print("  1. The robot is powered on")
            print("  2. PC and robot are on the same WiFi network")
            sys.exit(1)
        else:
            print(f"  Found {len(candidates)} candidate(s):")
            for i, c in enumerate(candidates):
                print(f"\n  [{i+1}] IP: {c['ip']}")
                print(f"       Confidence : {c['confidence']}")
                print(f"       Open ports : {c['open_ports']}")
                print(f"       MAC        : {c['mac'] or 'N/A'}")
                print(f"       Hostname   : {c['hostname'] or 'N/A'}")
                print(f"       Evidence   : {c['evidence']}")

            best = candidates[0]
            print(f"\n  Best candidate: {best['ip']}")

            stream_url = find_robot_stream_url(best['ip'])
            if stream_url:
                print(f"  Stream URL: {stream_url}")
                print(f"\n  Add to .env:")
                print(f"  ROBOT_IP={best['ip']}")
                print(f"  ROBOT_STREAM_URL={stream_url}")
