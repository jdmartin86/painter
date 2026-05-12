import AVFoundation
import UIKit
import CoreImage

// MARK: - Delegate

protocol CameraManagerDelegate: AnyObject {
    /// Called at most once per (1 / targetFPS) seconds on the main queue.
    func didCaptureFrame(_ jpegData: Data)
}

// MARK: - Manager

final class CameraManager: NSObject {

    weak var delegate: CameraManagerDelegate?

    // Target frame rate sent to the server (not the capture rate)
    var targetFPS: Double = 5.0

    // Resolution to resize frames to before encoding (keeps payload small)
    var outputSize = CGSize(width: 128, height: 128)

    // JPEG compression quality  0.0–1.0
    var jpegQuality: CGFloat = 0.6

    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.rlapp.camera.session")
    private let outputQueue = DispatchQueue(label: "com.rlapp.camera.output")
    private let ciContext = CIContext()

    private var lastSentTime: CFTimeInterval = 0

    // MARK: - Setup

    /// Call once, e.g. in viewDidLoad. Requests camera permission first.
    func setup(completion: @escaping (Result<Void, Error>) -> Void) {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            guard granted else {
                DispatchQueue.main.async {
                    completion(.failure(CameraError.permissionDenied))
                }
                return
            }
            self.sessionQueue.async {
                do {
                    try self.configureSession()
                    DispatchQueue.main.async { completion(.success(())) }
                } catch {
                    DispatchQueue.main.async { completion(.failure(error)) }
                }
            }
        }
    }

    private func configureSession() throws {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .medium  // 480p is plenty; we resize anyway

        // Input
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else {
            throw CameraError.deviceUnavailable
        }
        captureSession.addInput(input)

        // Output
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        guard captureSession.canAddOutput(videoOutput) else {
            throw CameraError.outputUnavailable
        }
        captureSession.addOutput(videoOutput)

        // Lock orientation to portrait (adjust if needed)
        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }

        captureSession.commitConfiguration()
    }

    // MARK: - Start / Stop

    func startCapture() {
        sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    func stopCapture() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
    }

    // MARK: - Preview layer

    /// Add the returned layer to your view's layer to show the live preview.
    func makePreviewLayer(for bounds: CGRect) -> AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.frame = bounds
        layer.videoGravity = .resizeAspectFill
        return layer
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Throttle to targetFPS
        let now = CACurrentMediaTime()
        let interval = 1.0 / targetFPS
        guard now - lastSentTime >= interval else { return }
        lastSentTime = now

        guard let jpegData = process(sampleBuffer) else { return }

        DispatchQueue.main.async { [weak self] in
            self?.delegate?.didCaptureFrame(jpegData)
        }
    }

    // MARK: - Frame processing

    private func process(_ sampleBuffer: CMSampleBuffer) -> Data? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Scale to outputSize
        let scaleX = outputSize.width  / ciImage.extent.width
        let scaleY = outputSize.height / ciImage.extent.height
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        // Render to CGImage
        guard let cgImage = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }

        // Encode as JPEG
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.jpegData(compressionQuality: jpegQuality)
    }
}

// MARK: - Errors

enum CameraError: LocalizedError {
    case permissionDenied
    case deviceUnavailable
    case outputUnavailable

    var errorDescription: String? {
        switch self {
        case .permissionDenied:  return "Camera permission denied."
        case .deviceUnavailable: return "No camera device available."
        case .outputUnavailable: return "Could not attach video output."
        }
    }
}
