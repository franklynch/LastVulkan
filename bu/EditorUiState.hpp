#pragma once

struct EditorUiState
{
    int selectedRenderableIndex = -1;

    bool showDebugPanel = true;
    bool showDemoWindow = false;

    bool wireframeRequested = false;
};