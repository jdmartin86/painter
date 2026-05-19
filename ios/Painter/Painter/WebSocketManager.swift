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

    private let encoder = JSONEncoder()

    override init() {
        super.init()
        urlSession = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    // MARK: - Connection Controls

    func connect(to urlString: String) {
        guard let url = URL(string: urlString) else {
            print("[WebSocket] Invalid URL layout target: \(urlString)")
            return
        }
        disconnect()
        print("[WebSocket] Opening pipe to: \(url.absoluteString)")
        webSocketTask = urlSession.webSocketTask(with: url)
        webSocketTask?.resume()
        scheduleReceive()
    }

    func disconnect() {
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        isConnected = false
    }

    // MARK: - Dual-Channel Transmissions (Saves the Loop)

    /// Transmits raw binary JPEG frames straight to the Python OpenCV processing array pipeline
    func sendFrame(_ jpegData: Data) {
        guard isConnected else { return }

        // FIX: Strip the JSON text wrapper and send pure binary over the network wire
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

        // Send as a clear string to trip Python's "text" parsing branch
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
            defer { self.scheduleReceive() } // Protect execution state loops
            
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
            case .failure(let error):
                print("[WebSocket] Connection line closed: \(error.localizedDescription)")
                if self.isConnected {
                    self.isConnected = false
                    DispatchQueue.main.async {
                        self.delegate?.didChangeConnectionState(false)
                    }
                }
            }
        }
    }

    private func decodeServerBinaryPayload(_ data: Data) {
        // Validation check based on your server's header requirements:
        // Python: struct.pack("!IBBf", ...) -> 4 bytes (UInt32) + 1 byte (UInt8) + 1 byte (UInt8) + 4 bytes (Float) = 10 bytes metadata header
        guard data.count > 10 else { return }
        
        // ── EXTRACT METADATA VIA BIG-ENDIAN UNPACKING ──
        // Step Count (Bytes 0-3)
        let step: UInt32 = data.subdata(in: 0..<4).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        
        // Grid Coordinates (Bytes 4 and 5)
        let row: UInt8 = data[4]
        let col: UInt8 = data[5]
        
        // Reward Float (Bytes 6-9)
        let rewardBits: UInt32 = data.subdata(in: 6..<10).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let reward = Float(bitPattern: rewardBits)
        
        // Extract remaining raw JPEG byte stream payload
        let jpegData = data.subdata(in: 10..<data.count)
        
        if let decodedImage = UIImage(data: jpegData) {
            DispatchQueue.main.async {
                // Return data elements smoothly back to the UI thread
                self.delegate?.didReceiveActionImage(decodedImage, step: step, row: row, col: col, reward: reward)
            }
        } else {
            print("[Parser] Failed to process image frame payload array sequence.")
        }
    }

    // MARK: - URLSessionWebSocketDelegate Protocols

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        isConnected = true
        DispatchQueue.main.async {
            self.delegate?.didChangeConnectionState(true)
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        isConnected = false
        DispatchQueue.main.async {
            self.delegate?.didChangeConnectionState(false)
        }
    }
}
