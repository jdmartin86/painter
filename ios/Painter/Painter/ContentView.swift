import SwiftUI
import AVFoundation
import Combine
import UIKit

// MARK: - ViewModel

final class RLViewModel: NSObject, ObservableObject { // NSObject inheritance required for photo saving selectors

    // UI state
    @Published var actionImage: UIImage? = nil
    @Published var isConnected: Bool = false
    @Published var statusMessage: String = "Not connected"
    @Published var framesSent: Int = 0
    
    // Alert Handling for Image Saving
    @Published var alertMessage: String? = nil
    @Published var showAlert: Bool = false

    // Sub-managers
    let cameraManager = CameraManager()
    let wsManager = WebSocketManager()

    // Change this to your server's address
    private let serverURL = "ws://192.168.86.249:8000/ws" // Me
    //    private let serverURL = "ws://192.168.7.36:8000/ws" // Will's

    override init() {
        super.init()
        cameraManager.delegate = self
        wsManager.delegate = self
    }

    // MARK: - Lifecycle

    func onAppear() {
        UIApplication.shared.isIdleTimerDisabled = true
        cameraManager.setup { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success:
                self.statusMessage = "Establishing connection..."
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

    // MARK: - Core Actions

    func sendReward(_ value: Float) {
        wsManager.sendReward(value)
    }
    
    func saveCurrentImage() {
        guard let imageToSave = actionImage else {
            triggerAlert(message: "No frame available to save yet.")
            return
        }
        
        // Targets the photo library utilizing Objective-C dynamic dispatch
        UIImageWriteToSavedPhotosAlbum(imageToSave, self, #selector(saveImageCallback(_:didFinishSavingWithError:contextInfo:)), nil)
    }
    
    @objc private func saveImageCallback(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            triggerAlert(message: "Save failed: \(error.localizedDescription)")
        } else {
            triggerAlert(message: "Frame saved!")
        }
    }
    
    private func triggerAlert(message: String) {
        DispatchQueue.main.async {
            self.alertMessage = message
            self.showAlert = true
        }
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
    func didReceiveActionImage(_ image: UIImage, step: UInt32, row: UInt8, col: UInt8, reward: Float) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.actionImage = image
        }
    }

    func didChangeConnectionState(_ connected: Bool) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            guard self.isConnected != connected else { return }
            self.isConnected = connected
            
            if connected {
                self.statusMessage = "Connected"
                self.cameraManager.startCapture()
            } else {
                self.statusMessage = "Connection lost. Retrying in 2s..."
                self.cameraManager.stopCapture()
                
                // Controlled retry loop schedule
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
                    guard let self = self else { return }
                    // Double check we didn't get reconnected via delegate in the meantime
                    guard !self.isConnected else { return }
                    
                    print("[UI] Attempting scheduled reconnection...")
                    self.wsManager.connect(to: self.serverURL)
                }
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
            Color.black.ignoresSafeArea()

            if viewModel.isConnected {
                // ── ACTIVE SESSION INTERFACE ──
                Group {
                    if let uiImage = viewModel.actionImage {
                        Image(uiImage: uiImage)
                            .interpolation(.none)
                            .resizable()
                            .scaledToFit()
                    } else {
                        CameraPreview(viewModel: viewModel)
                            .ignoresSafeArea()
                    }
                }
                
                // HUD Overlay (Only visible when connected)
                VStack {
                    HStack {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(Color.green)
                                .frame(width: 8, height: 8)
                            Text(viewModel.statusMessage)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.white)
                                .shadow(radius: 2)
                        }
                        Spacer()
                        Text("Frames: \(viewModel.framesSent)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.white)
                            .shadow(radius: 2)
                    }
                    .padding()

                    Spacer()

                    // Controls Row
                    HStack(spacing: 40) {
                        Button(action: { viewModel.saveCurrentImage() }) {
                            Image(systemName: "square.and.arrow.down")
                                .font(.system(size: 24, weight: .semibold))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Color.white.opacity(0.2), in: Circle())
                        }
                        .buttonStyle(.plain)

                        RewardButton(label: "➕", color: .green) {
                            viewModel.sendReward(+1.0)
                        }
                    }
                    .padding(.bottom, 30)
                }
            } else {
                // ── STARTUP / RECONNECTING SCREEN ──
                VStack(spacing: 20) {
                    Spacer()
                    
                    // App Network Status Icon
                    Image(systemName: "pencil.and.outline")
                            .font(.system(size: 72)) // Slightly larger looks great for this specific symbol
                            .foregroundColor(.white)  // A crisp blue theme, or use .white / .accentColor
                            .symbolEffect(.bounce, options: .repeating) // Creates an active drawing/bouncing effect
                    
                    VStack(spacing: 8) {
                        Text("Connecting to Agent")
                            .font(.system(.title2, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                    }
                    Spacer()
                }
                .transition(.opacity.combined(with: .scale(scale: 0.95)))
            }
        }
        .animation(.easeInOut(duration: 0.4), value: viewModel.isConnected) // Seamless swap transition
        .preferredColorScheme(.dark)
        .onAppear { viewModel.onAppear() }
        .onDisappear { viewModel.onDisappear() }
        .alert("Photo Library", isPresented: $viewModel.showAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(viewModel.alertMessage ?? "")
        }
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
            withAnimation(.easeOut(duration: 0.1)) { pressed = true }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                withAnimation { pressed = false }
            }
            action()
        }) {
            Text(label)
                .font(.system(size: 32))
                .frame(width: 60, height: 60)
                .background(color, in: Circle())
                .scaleEffect(pressed ? 0.92 : 1.0)
                .brightness(pressed ? 0.2 : 0)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
