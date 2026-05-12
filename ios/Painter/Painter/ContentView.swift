import SwiftUI
import AVFoundation
import Combine

// MARK: - ViewModel

final class RLViewModel: ObservableObject {

    // UI state
    @Published var actionText: String = "—"
    @Published var isConnected: Bool = false
    @Published var statusMessage: String = "Not connected"
    @Published var framesSent: Int = 0

    // Sub-managers
    private let cameraManager = CameraManager()
    private let wsManager = WebSocketManager()

    // Change this to your server's address
    private let serverURL = "ws://192.168.86.249:8000/ws"

    init() {
        cameraManager.delegate = self
        wsManager.delegate = self
    }

    // MARK: - Lifecycle

    func onAppear() {
        cameraManager.setup { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.cameraManager.startCapture()
                self.wsManager.connect(to: self.serverURL)
            case .failure(let error):
                DispatchQueue.main.async {
                    self.statusMessage = error.localizedDescription
                }
            }
        }
    }

    func onDisappear() {
        cameraManager.stopCapture()
        wsManager.disconnect()
    }

    // MARK: - Camera preview

    func makePreviewLayer(bounds: CGRect) -> AVCaptureVideoPreviewLayer {
        cameraManager.makePreviewLayer(for: bounds)
    }

    // MARK: - Reward buttons

    func sendReward(_ value: Float) {
        wsManager.sendReward(value)
    }
}

// MARK: - CameraManagerDelegate

extension RLViewModel: CameraManagerDelegate {
    func didCaptureFrame(_ jpegData: Data) {
        wsManager.sendFrame(jpegData)
        DispatchQueue.main.async {
            self.framesSent += 1
        }
    }
}

// MARK: - WebSocketManagerDelegate

extension RLViewModel: WebSocketManagerDelegate {
    func didReceiveAction(_ action: ActionMessage) {
        actionText = action.action
    }

    func didChangeConnectionState(_ connected: Bool) {
        isConnected = connected
        statusMessage = connected ? "Connected" : "Disconnected — retrying…"
        if !connected {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
                guard let self else { return }
                self.wsManager.connect(to: self.serverURL)
            }
        }
    }
}

// MARK: - Camera Preview (UIViewRepresentable)

struct CameraPreview: UIViewRepresentable {
    let viewModel: RLViewModel

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        // Add preview layer once the view has a non-zero frame
        guard uiView.bounds != .zero,
              uiView.layer.sublayers?.contains(where: { $0 is AVCaptureVideoPreviewLayer }) == false
        else { return }

        let previewLayer = viewModel.makePreviewLayer(bounds: uiView.bounds)
        uiView.layer.insertSublayer(previewLayer, at: 0)
    }
}

// MARK: - ContentView

struct ContentView: View {

    @StateObject private var viewModel = RLViewModel()

    var body: some View {
        ZStack {

            // ── Camera feed (full screen background) ──
            CameraPreview(viewModel: viewModel)
                .ignoresSafeArea()

            // ── HUD overlay ──
            VStack {

                // Status bar
                HStack {
                    Circle()
                        .fill(viewModel.isConnected ? Color.green : Color.red)
                        .frame(width: 10, height: 10)
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.white)
                    Spacer()
                    Text("Frames: \(viewModel.framesSent)")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.7))
                }
                .padding(.horizontal)
                .padding(.top, 8)
                .background(.black.opacity(0.4))

                Spacer()

                // Agent action display
                VStack(spacing: 4) {
                    Text("Agent Action")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.7))
                    Text(viewModel.actionText)
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                }
                .padding(.bottom, 24)

                // Reward buttons
                HStack(spacing: 32) {
//                    RewardButton(label: "👎", color: .red) {
//                        viewModel.sendReward(-1.0)
//                    }
                    RewardButton(label: "➕", color: .green) {
                        viewModel.sendReward(+1.0)
                    }
                }
                .padding(.bottom, 48)
            }
        }
        .onAppear  { viewModel.onAppear() }
        .onDisappear { viewModel.onDisappear() }
    }
}

// MARK: - RewardButton

struct RewardButton: View {
    let label: String
    let color: Color
    let action: () -> Void

    @State private var pressed = false

    var body: some View {
        Button(action: {
            withAnimation(.easeOut(duration: 0.15)) { pressed = true }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                withAnimation { pressed = false }
            }
            action()
        }) {
            Text(label)
                .font(.system(size: 36))
                .frame(width: 72, height: 72)
                .background(color.opacity(pressed ? 0.9 : 0.5),
                            in: Circle())
                .scaleEffect(pressed ? 0.92 : 1.0)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
