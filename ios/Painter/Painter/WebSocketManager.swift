import Foundation
import UIKit

// MARK: - Outbound Message Types

struct RewardMessage: Encodable {
    let type = "reward"
    let value: Float          // e.g. +1.0 or -1.0
    let timestamp: Double
}

// MARK: - Delegate Protocol

protocol WebSocketManagerDelegate: AnyObject {
    func didReceiveActionImage(_ image: UIImage, step: UInt32, row: UInt8, col: UInt8, reward: Float)
    func didChangeConnectionState(_ connected: Bool)
}

// MARK: - WebSocket Manager

final class WebSocketManager: NSObject, URLSessionWebSocketDelegate {

    weak var delegate: WebSocketManagerDelegate?

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession!
    private(set) var isConnected = false
    
    // ── Reconnection & Concurrency Safeguards ─────────────────────────────────
    private var isConnecting = false
    private var lastTargetURLString: String?
    private var reconnectDelay: TimeInterval = 1.0
    private let maxReconnectDelay: TimeInterval = 16.0
    private var isIntentionallyDisconnected = false

    private let encoder = JSONEncoder()

    override init() {
        super.init()
        // Utilizing a dedicated serial operation queue context for underlying socket stability
        urlSession = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    // MARK: - Connection Controls

    func connect(to urlString: String) {
        guard !isConnected && !isConnecting else {
            print("[WebSocket] Connection attempt ignored: isConnected=\(isConnected), isConnecting=\(isConnecting)")
            return
        }
        
        guard let url = URL(string: urlString) else {
            print("[WebSocket] Invalid URL layout target: \(urlString)")
            return
        }
        
        lastTargetURLString = urlString
        isIntentionallyDisconnected = false
        isConnecting = true
        
        // Ensure old tasks are aggressively scrubbed before starting a fresh run
        disconnect(intentional: false)
        
        print("[WebSocket] Opening pipe to: \(url.absoluteString)")
        webSocketTask = urlSession.webSocketTask(with: url)
        webSocketTask?.resume()
        scheduleReceive()
    }

    /// Public teardown connection request
    func disconnect() {
        disconnect(intentional: true)
    }
    
    private func disconnect(intentional: Bool) {
        isIntentionallyDisconnected = intentional
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        
        if isConnected {
            isConnected = false
            DispatchQueue.main.async {
                self.delegate?.didChangeConnectionState(false)
            }
        }
        
        if intentional {
            isConnecting = false
        }
    }
    
    // MARK: - Exponential Backoff Reconnection Logic
    
    private func handleDisruption() {
        guard !isIntentionallyDisconnected, let urlString = lastTargetURLString else { return }
        
        // Prevent stacking duplicate reconnect execution timers
        guard !isConnecting else { return }
        
        disconnect(intentional: false)
        isConnecting = true
        
        print("[WebSocket] Connection dropped. Retrying link in \(reconnectDelay)s...")
        
        DispatchQueue.global().asyncAfter(deadline: .now() + reconnectDelay) { [weak self] in
            guard let self = self else { return }
            
            // Advance exponential backoff tracking spacing
            self.reconnectDelay = min(self.reconnectDelay * 2, self.maxReconnectDelay)
            
            // Clear flag strictly right before attempting connect so guard check can pass
            self.isConnecting = false
            self.connect(to: urlString)
        }
    }

    // MARK: - Dual-Channel Transmissions (Saves the Loop)

    /// Transmits raw binary JPEG frames straight to the Python OpenCV processing array pipeline
    func sendFrame(_ jpegData: Data) {
        guard isConnected else { return }

        webSocketTask?.send(.data(jpegData)) { error in
            if let error = error {
                print("[WebSocket] Binary frame send error: \(error.localizedDescription)")
            }
        }
    }

    /// Transmits user reward interaction frames wrapped cleanly as JSON text blocks
    func sendReward(_ value: Float) {
        guard isConnected else { return }

        let message = RewardMessage(
            value: value,
            timestamp: Date().timeIntervalSince1970
        )
        
        guard let jsonData = try? encoder.encode(message),
              let jsonString = String(data: jsonData, encoding: .utf8) else { return }

        webSocketTask?.send(.string(jsonString)) { error in
            if let error = error {
                print("[WebSocket] Reward text send error: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Receiving Loop Processing

    private func scheduleReceive() {
        guard let task = webSocketTask else { return }
        
        task.receive { [weak self] result in
            guard let self = self else { return }
            
            switch result {
            case .success(let message):
                switch message {
                case .data(let data):
                    self.decodeServerBinaryPayload(data)
                case .string(let text):
                    print("[WebSocket] Received text packet frame back from server: \(text)")
                @unknown default:
                    break
                }
                // Keep the loop running seamlessly
                self.scheduleReceive()
                
            case .failure(let error):
                print("[WebSocket] Loop read error context: \(error.localizedDescription)")
                self.handleDisruption()
            }
        }
    }

    private func decodeServerBinaryPayload(_ data: Data) {
        // Expected payload structure size minimum check:
        // !IBBf -> 4 bytes (UInt32) + 1 byte (UInt8) + 1 byte (UInt8) + 4 bytes (Float) = 10 bytes metadata header
        guard data.count > 10 else { return }
        
        // Step Count (Bytes 0-3)
        let step: UInt32 = data.subdata(in: 0..<4).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        
        // Grid Coordinates (Bytes 4 and 5)
        let row: UInt8 = data[4]
        let col: UInt8 = data[5]
        
        // Reward Float (Bytes 6-9)
        let rewardBits: UInt32 = data.subdata(in: 6..<10).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let reward = Float(bitPattern: rewardBits)
        
        // Extract trailing raw JPEG image payload data
        let jpegData = data.subdata(in: 10..<data.count)
        
        if let decodedImage = UIImage(data: jpegData) {
            DispatchQueue.main.async {
                self.delegate?.didReceiveActionImage(decodedImage, step: step, row: row, col: col, reward: reward)
            }
        } else {
            print("[Parser] Failed to process image frame payload array sequence.")
        }
    }

    // MARK: - URLSessionWebSocketDelegate Protocols

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        print("[WebSocket] Socket line open and confirmed active.")
        isConnected = true
        isConnecting = false
        reconnectDelay = 1.0 // Reset delay penalty on active connection validation
        
        DispatchQueue.main.async {
            self.delegate?.didChangeConnectionState(true)
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        print("[WebSocket] Server triggered a clean close routine.")
        isConnecting = false
        if closeCode == .goingAway && isIntentionallyDisconnected {
            isConnected = false
            DispatchQueue.main.async {
                self.delegate?.didChangeConnectionState(false)
            }
        } else {
            handleDisruption()
        }
    }
    
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("[WebSocket] Session task transport failure context: \(error.localizedDescription)")
            isConnecting = false
            handleDisruption()
        }
    }
}
