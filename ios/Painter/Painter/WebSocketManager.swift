import Foundation

// MARK: - Message types

struct FrameMessage: Encodable {
    let type = "frame"
    let data: String          // base64-encoded JPEG
    let timestamp: Double
}

struct RewardMessage: Encodable {
    let type = "reward"
    let value: Float          // e.g. +1.0 or -1.0
    let timestamp: Double
}

struct ActionMessage: Decodable {
    let type: String          // TODO: This should be an image too.
    let action: String        // display string from the agent
    let value: [Float]?       // optional raw action vector
}

// MARK: - Delegate

protocol WebSocketManagerDelegate: AnyObject {
    func didReceiveAction(_ action: ActionMessage)
    func didChangeConnectionState(_ connected: Bool)
}

// MARK: - Manager

final class WebSocketManager: NSObject {

    weak var delegate: WebSocketManagerDelegate?

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession!
    private(set) var isConnected = false

    private let encoder = JSONEncoder()

    override init() {
        super.init()
        urlSession = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    // MARK: - Connection

    func connect(to urlString: String) {
        guard let url = URL(string: urlString) else {
            print("WebSocketManager: invalid URL — \(urlString)")
            return
        }
        disconnect()
        webSocketTask = urlSession.webSocketTask(with: url)
        webSocketTask?.resume()
        scheduleReceive()
    }

    func disconnect() {
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        isConnected = false
    }

    // MARK: - Sending

    /// Call this with a compressed JPEG Data object (already resized to target resolution).
    func sendFrame(_ jpegData: Data, timestamp: Double = Date().timeIntervalSince1970) {
        guard isConnected else { return }

        let message = FrameMessage(
            data: jpegData.base64EncodedString(),
            timestamp: timestamp
        )
        send(encodable: message)
    }

    func sendReward(_ value: Float) {
        guard isConnected else { return }

        let message = RewardMessage(
            value: value,
            timestamp: Date().timeIntervalSince1970
        )
        send(encodable: message)
    }

    // MARK: - Receiving

    private func scheduleReceive() {
        webSocketTask?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                self.handleMessage(message)
                self.scheduleReceive()          // keep the receive loop alive
            case .failure(let error):
                print("WebSocketManager receive error: \(error.localizedDescription)")
                self.isConnected = false
                self.delegate?.didChangeConnectionState(false)
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            guard let data = text.data(using: .utf8) else { return }
            decode(data)
        case .data(let data):
            decode(data)
        @unknown default:
            break
        }
    }

    private func decode(_ data: Data) {
        do {
            let action = try JSONDecoder().decode(ActionMessage.self, from: data)
            DispatchQueue.main.async {
                self.delegate?.didReceiveAction(action)
            }
        } catch {
            print("WebSocketManager decode error: \(error)")
        }
    }

    // MARK: - Helpers

    private func send(encodable: some Encodable) {
        guard let data = try? encoder.encode(encodable),
              let text = String(data: data, encoding: .utf8) else { return }

        webSocketTask?.send(.string(text)) { error in
            if let error {
                print("WebSocketManager send error: \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - URLSessionWebSocketDelegate

extension WebSocketManager: URLSessionWebSocketDelegate {

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol protocol: String?
    ) {
        isConnected = true
        DispatchQueue.main.async {
            self.delegate?.didChangeConnectionState(true)
        }
        print("WebSocketManager: connected")
    }

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        isConnected = false
        DispatchQueue.main.async {
            self.delegate?.didChangeConnectionState(false)
        }
        print("WebSocketManager: disconnected (\(closeCode))")
    }
}
