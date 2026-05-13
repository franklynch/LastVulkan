#pragma once

struct InputState
{
    bool wantsMouseCapture = false;

    bool rightMouseDown = false;
    bool middleMouseDown = false;

    float mouseDeltaX = 0.0f;
    float mouseDeltaY = 0.0f;

    float scrollDelta = 0.0f;
};